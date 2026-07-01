"""
GUIAgent —— Vision-grounded Android GUI Agent

主循环：
    user_task
        ↓
    for step in range(max_steps):
        screenshot = adb.screencap        # 真机截屏
        decision = vlm.chat(prompt, image=screenshot)
                                          # VLM 直接吐坐标
                                          # {"action":"tap","x":540,"y":1200,...}
        executor.do(decision)             # adb 物理像素 tap/type/swipe
        if decision.done: break
    return GUIAgentResult(trace, success)

设计要点：
- 截图缩放到 ≤1280 宽再发给模型（省 token），坐标按比例缩放回设备分辨率
- VLM 看到的就是用户看到的，不再走 UI 树（UI 树只留给 skill 自愈兜底）
- 输出严格 JSON：{thought, action, x, y, text?, direction?, package?, done}
- 失败、坐标越界都按"动作失败"处理，下一步让 LLM 自己看新屏幕重决策
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mobilerun.executor import AdbExecutor
from mobilerun.llm import BaseLLM


_SYSTEM = """你是一个操作 Android 手机的 GUI Agent。看屏幕截图 + 任务 → 输出**下一步**动作。

# 输出规范（最重要，违反一次任务就挂）
**只输出一个 JSON 对象**。不要 Markdown 代码块、不要 ```json、不要前后任何说明文字。
JSON 必须 100% 合法：
- 字符串两端必须配对的双引号，数字两侧**禁止**出现引号
  错：{{"y": 1105"}}    对：{{"y": 1105}}
- tap / long_press 必须**同时**给 "x" 和 "y" 两个键，**不能**写成 {{"x": 405, 1107}}
- 不能有末尾逗号、不能用单引号、不能用 Python 的 True/False/None

正确示例（你要照这个格式输出，但内容根据屏幕决定；x/y 在 [0,1000] 区间）：
{{"action": "tap", "x": 900, "y": 70, "thought": "点击右上角搜索图标", "done": false}}

# 可用动作（每次只能选一个）
- {{"action": "tap",       "x": <int>, "y": <int>, "thought": "...", "done": false}}
- {{"action": "long_press","x": <int>, "y": <int>, "thought": "...", "done": false}}
- {{"action": "type",      "text": "<要输入的文字>", "thought": "...", "done": false}}
       （想往输入框/搜索框打字时直接用 type。**本机输入法不会弹出可见键盘**，
        所以**别等键盘出现、别因为"没看到键盘"就反复点输入框**：
        最多点选输入框一次，**下一步立刻 type**。
        若输入框里已有文字要先清空，加 "clear": true，例：
        {{"action": "type", "text": "北京", "clear": true, "done": false}}）
- {{"action": "enter",     "thought": "提交搜索/确认输入（等同回车）", "done": false}}
- {{"action": "clear",     "thought": "清空当前输入框", "done": false}}
- {{"action": "swipe",     "direction": "up|down|left|right", "thought": "...", "done": false}}
- {{"action": "back",      "thought": "...", "done": false}}
- {{"action": "open_app",  "package": "com.tencent.mm", "thought": "...", "done": false}}
- {{"action": "wait",      "thought": "等页面加载", "done": false}}
- {{"action": "finish",    "thought": "任务完成", "done": true}}

# 坐标说明（极重要，写错就点不到）
- x, y 一律使用 **[0, 1000] 归一化坐标**（图像左上角 = (0,0)，右下角 = (1000,1000)）
  例：屏幕右上角的搜索图标，x ≈ 900~950，y ≈ 50~100
  例：屏幕正中间的元素，x ≈ 500，y ≈ 500
  例：屏幕底部 tab 栏的图标，y ≈ 950
- **不要**输出原始像素坐标，不要超过 1000
- 当前截图原始尺寸供你参考：宽 {img_w} × 高 {img_h}（**只用于估算比例**，输出仍是 [0,1000]）
- 系统会自动把归一化坐标缩放到设备物理像素并 tap

# 决策原则（避免常见坑）
1. **先看清当前屏幕在哪个 app / 哪个页面**（看顶部标题、底部 tab、键盘有没有弹出）
2. **若已经在目标页面或目标元素已可见，直接 tap / type，不要乱按 back**
   —— 例如打开微信后默认在"聊天"列表，要找联系人就**用搜索**，不要按 back 退出微信
3. **找联系人/应用/设置项的标准流程：用搜索，不要猜列表位置**
   微信：打开微信 → tap 右上角放大镜图标（≈ 截图右上角，y 大约在顶栏内）→
        输入名字 → tap 搜索结果里"联系人"分区的目标条目 → 进入聊天页 →
        tap 底部输入框 → type 消息 → tap"发送"按钮
   **不要在聊天列表里凭印象 tap 第几条**——你看不清谁是谁的时候必须走搜索
4. tap 的坐标要指向元素的**中心**，避免误点边缘或临近控件
5. **输入文字**：点选输入框后**立刻 type**；本机输入法不弹可见键盘，
   看到顶部有搜索框/输入框就直接 type，**不要反复点同一个输入框**
6. **如果上一步动作执行后页面没有变化或进入了错的页面**（参考"已经做过的步骤"），
   说明刚才那一下没生效 —— **换完全不同的策略**：刚才在点输入框就改用 type 直接输入；
   否则换元素/换路径，**不要重复同样的 tap**
7. 只要任务**已经完成**（消息已发送 / 设置已开启 / 备忘录已保存），立刻输出 finish + done=true
"""


_USER_TEMPLATE = """## 用户任务
{task}

## 已经做过的步骤
{history}

请观察截图，输出下一步的 JSON。
"""


# Set-of-Marks 提示块：仅当无障碍树非空时才拼到 prompt 里
_SOM_BLOCK = """# 本屏可交互元素（已在截图上用红框 + 编号标注，编号在框左上角）
{elements}

# 选择上面任意元素时，**优先**用编号点击（最准，系统会点该元素中心）：
#   {{"action": "tap_id", "id": <编号int>, "thought": "...", "done": false}}
# 输入文字到某个输入框：先 tap_id 选中它，下一步再 type。
# 只有当目标**没出现在上面编号列表里**（画布 / 游戏 / 网页内元素）时，
# 才回退用 {{"action": "tap", "x": <int>, "y": <int>, ...}} 自估坐标。"""


# 不同 VLM 对同一动作用词不一，统一映射到本 agent 的动作词汇
_ACTION_ALIASES = {
    "click": "tap", "left_click": "tap", "press": "tap", "touch": "tap",
    "click_id": "tap_id", "tapid": "tap_id",
    "long_click": "long_press", "longpress": "long_press", "long_tap": "long_press",
    "input": "type", "input_text": "type", "enter_text": "type",
    "type_text": "type", "settext": "type",
    "scroll": "swipe",
    "go_back": "back", "navigate_back": "back", "system_back": "back", "press_back": "back",
    "done": "finish", "complete": "finish", "completed": "finish", "stop": "finish",
    "launch_app": "open_app", "open": "open_app", "start_app": "open_app", "launch": "open_app",
    "enter": "enter", "press_enter": "enter", "keyboard_enter": "enter",
    "submit": "enter", "search": "enter", "go": "enter",
    "clear": "clear", "clear_text": "clear", "clear_field": "clear",
}

# 不可逆 / 外发 / 花钱类操作的关键词。命中时会告警；开启 block_risky_actions
# 后会在执行前拦截（避免自主 agent 误删/误购/误发）。
_RISKY_KEYWORDS = (
    "卸载", "uninstall", "删除", "delete", "remove account", "移除",
    "支付", "付款", "pay", "购买", "buy", "purchase", "下单", "subscribe", "订阅",
    "发送", "send", "格式化", "factory reset", "恢复出厂", "erase", "清除数据",
    "confirm purchase", "place order", "立即购买", "立即支付",
)


def _is_risky_label(label: str) -> bool:
    s = (label or "").strip().lower()
    return bool(s) and any(k in s for k in _RISKY_KEYWORDS)


@dataclass
class GUIAgentStep:
    step_index: int
    thought: str
    action: Dict[str, Any]
    ok: bool
    error: str = ""


@dataclass
class GUIAgentResult:
    task: str
    success: bool
    steps: List[GUIAgentStep] = field(default_factory=list)
    stop_reason: str = ""

    def trace_for_skill(self) -> List[Dict[str, Any]]:
        """成功路径上的步骤序列（去掉失败步和 finish），可喂给 SkillExtractor。

        注意：坐标用执行时换算出的**设备像素**（action["_device_xy"]），
        而不是模型输出的 [0,1000] 归一化值 —— 这样 SkillRunner 在
        同分辨率设备上回放时才能直接 `input tap`。动作名也从 agent 的
        `tap` 归一到技能库词汇 `click`。
        """
        out = []
        for s in self.steps:
            if not s.ok:
                continue
            a = s.action
            t = a.get("action")
            if t == "finish":
                continue
            if t in ("tap", "tap_id"):
                t = "click"
            entry = {"action_type": t}
            for k in ("text", "direction", "package"):
                if a.get(k):
                    entry[k] = a[k]
            if a.get("_device_xy"):
                entry["coordinates"] = [int(a["_device_xy"][0]),
                                        int(a["_device_xy"][1])]
            elif "x" in a and "y" in a:
                entry["coordinates"] = [int(a["x"]), int(a["y"])]
            out.append(entry)
        return out


class GUIAgent:
    """Vision-grounded GUI Agent.

    用法：
        from mobilerun.executor import AdbExecutor
        from mobilerun.llm import build_llm_from_env
        from mobilerun.agent import GUIAgent

        agent = GUIAgent(AdbExecutor(), build_llm_from_env(vision=True))
        result = agent.run("打开微信，给张三发『我到了』")
    """

    def __init__(
        self,
        executor: AdbExecutor,
        llm: BaseLLM,
        *,
        max_steps: int = 20,
        screenshot_max_width: int = 1280,
        loop_threshold: int = 3,
        screens_dir: Optional[str] = None,
        use_som: bool = True,
        som_max_marks: int = 50,
        block_risky_actions: bool = False,
        max_waits: int = 40,
        max_consecutive_waits: int = 25,
        wait_base_seconds: float = 1.5,
        wait_cap_seconds: float = 8.0,
        log=None,
    ):
        self.executor = executor
        self.llm = llm
        self.max_steps = max_steps
        self.screenshot_max_width = screenshot_max_width
        self.loop_threshold = loop_threshold
        # wait 不消耗 max_steps（动作预算），单独设上限；连续 wait 睡眠递增，
        # 这样等大文件下载/长加载时既不挤占动作步数，也大幅减少 VLM 调用次数。
        self.max_waits = max_waits
        self.max_consecutive_waits = max_consecutive_waits
        self.wait_base_seconds = wait_base_seconds
        self.wait_cap_seconds = wait_cap_seconds
        self.screens_dir = Path(screens_dir) if screens_dir else None
        if self.screens_dir:
            self.screens_dir.mkdir(parents=True, exist_ok=True)
        # Set-of-Marks：用无障碍树给可点元素叠编号，模型按编号选，点元素中心
        # （比模型自估坐标准）。树为空时自动退回纯视觉坐标模式。
        self.use_som = use_som
        self.som_max_marks = som_max_marks
        # 风险动作护栏：命中危险关键词时告警；开启后在执行前直接拦截
        self.block_risky_actions = block_risky_actions
        self._cur_elements: List[Any] = []
        self.log = log or print

    # ------------------------------------------------------------------
    def run(self, task: str) -> GUIAgentResult:
        self.log(f"\n=== Task: {task} ===")
        result = GUIAgentResult(task=task, success=False)
        history: List[str] = []
        recent_actions: List[str] = []
        last_frame_hash: Optional[str] = None
        stagnant_count = 0
        prev_action_type: Optional[str] = None

        # 每次 run 单独建带时间戳的截图子目录，避免覆盖上一次的记录
        run_screens = self.screens_dir
        if self.screens_dir:
            run_screens = self.screens_dir / time.strftime("run_%Y%m%d_%H%M%S")
            run_screens.mkdir(parents=True, exist_ok=True)

        step_i = 0            # 单调递增，仅用于日志/截图命名
        action_budget = 0     # 只统计真实交互动作；wait 不计入，受 max_steps 限制
        wait_total = 0
        consecutive_waits = 0

        while action_budget < self.max_steps:
            step_i += 1
            self.log(f"\n--- Step {step_i} ---")

            # 1) 观察：截图 + 缩放
            try:
                raw_png = self.executor.screenshot()
            except Exception as e:
                result.stop_reason = f"screenshot failed: {e}"
                return result

            small_png, (img_w, img_h), scale = self._resize_png(raw_png)
            self.log(f"[Observe] image {img_w}x{img_h}  scale={scale:.3f}")

            # 帧变化检测：上一步动作执行后屏幕到底有没有变（用未标注原图算哈希）。
            # 连续多次"动作后屏幕不动"基本等于卡死 / grounding 错了。
            fh = self._frame_hash(small_png)
            if (fh is not None and fh == last_frame_hash
                    and prev_action_type not in (None, "wait")):
                stagnant_count += 1
                hint = "(上一步执行后屏幕无明显变化，请换一个策略)"
                if prev_action_type in ("tap", "tap_id"):
                    hint = ("(上一步 tap 后屏幕没变：如果你点的是输入框，"
                            "现在请直接用 type 输入文字，别再点它；否则换元素/换路径)")
                history.append(hint)
                history = history[-10:]
            else:
                stagnant_count = 0
            last_frame_hash = fh

            if stagnant_count >= self.loop_threshold:
                result.stop_reason = (
                    f"detected stall: 连续 {self.loop_threshold} 次动作后屏幕无变化"
                )
                self.log(f"=== Stop: {result.stop_reason} ===")
                self.log("    提示：连续操作但页面都没动，多半 grounding 错了或卡住。"
                         "看 data/screens/ 对照。")
                return result

            # Set-of-Marks：抓无障碍树，给可点元素叠红框+编号。树空则退回纯视觉。
            elements: List[Any] = []
            vlm_png = small_png
            if self.use_som:
                elements = self._collect_elements()
                if elements:
                    vlm_png = self._annotate(small_png, elements, scale)
                    self.log(f"[SoM]     标注 {len(elements)} 个可交互元素")
            self._cur_elements = elements

            # 截图落盘存"模型实际看到的那张"（带标注），方便复盘
            if run_screens:
                p = run_screens / f"step_{step_i:02d}.png"
                p.write_bytes(vlm_png)
                self.log(f"[Observe] 已保存截图 {p}")

            # 2) 问 VLM（允许一次"只返回 JSON"的重试）
            prompt = self._build_prompt(task, history, img_w, img_h, elements)
            decision = None
            rsp = ""
            for attempt in range(2):
                try:
                    p = prompt if attempt == 0 else (
                        prompt + "\n\n上次回复不是合法 JSON，请**只**输出一个 JSON 对象，"
                        "不要任何代码块/前后文字。"
                    )
                    rsp = self.llm.chat(p, image=vlm_png)
                except Exception as e:
                    step = GUIAgentStep(step_i, "", {}, False, f"LLM error: {e}")
                    result.steps.append(step)
                    result.stop_reason = f"llm error: {e}"
                    return result
                decision = self._parse(rsp)
                if decision is not None:
                    break
                self.log(f"[Warn]    解析失败，原文: {rsp[:200]!r} -- 重试")

            if decision is None:
                self.log(f"[Fail]    模型连续返回非 JSON。最后原文:\n{rsp[:400]}")
                step = GUIAgentStep(step_i, "", {}, False,
                                    f"non-JSON: {rsp[:200]}")
                result.steps.append(step)
                result.stop_reason = "non-JSON response"
                return result

            thought = decision.get("thought", "")
            action = self._normalize_action(decision)
            self.log(f"[Plan]    {thought[:80]}")
            self.log(f"[Plan]    {action}")

            # 3) 结束判定
            if action.get("action") == "finish" or decision.get("done"):
                # 模型有时在 done=true 的同时还给了一个真实动作（如最后那次
                # "发送" tap）。先把这步执行掉，别让"宣称完成但没做最后一步"
                # 直接算成功。
                act_type = action.get("action")
                if act_type and act_type != "finish":
                    ok, err = self._dispatch(action, scale)
                    self.log(f"[Act]     {'OK' if ok else 'FAIL'} {err} (final)")
                    step = GUIAgentStep(step_i, thought, action, ok, err)
                else:
                    step = GUIAgentStep(step_i, thought, action, True, "")
                result.steps.append(step)
                result.success = True
                result.stop_reason = "done"
                self.log(f"=== Done after {step_i} steps ===")
                return result

            # 3.5) wait 特判：不消耗动作预算；连续 wait 睡眠递增、单独设上限。
            #      （等大文件下载/长加载时别白烧动作步数和 VLM 调用）
            if action.get("action") == "wait":
                wait_total += 1
                consecutive_waits += 1
                dur = min(self.wait_base_seconds * consecutive_waits,
                          self.wait_cap_seconds)
                self.log(f"[Act]     wait {dur:.1f}s "
                         f"(连续第 {consecutive_waits} 次，不计入步数)")
                step = GUIAgentStep(step_i, thought, action, True, f"wait {dur:.1f}s")
                result.steps.append(step)
                history.append(self._summarize_step(action, True))
                history = history[-10:]
                prev_action_type = "wait"
                recent_actions.clear()
                if (wait_total >= self.max_waits
                        or consecutive_waits >= self.max_consecutive_waits):
                    result.stop_reason = (
                        f"等待超时：连续 {consecutive_waits} 次 / 累计 {wait_total} 次 wait"
                    )
                    self.log(f"=== Stop: {result.stop_reason} ===")
                    return result
                time.sleep(dur)
                continue

            # 4) 执行（真实交互动作，消耗一格动作预算）
            ok, err = self._dispatch(action, scale)
            self.log(f"[Act]     {'OK' if ok else 'FAIL'} {err}")
            step = GUIAgentStep(step_i, thought, action, ok, err)
            result.steps.append(step)
            history.append(self._summarize_step(action, ok))
            history = history[-10:]
            prev_action_type = action.get("action")
            action_budget += 1
            consecutive_waits = 0

            # 5) 死循环检测：只对 tap / tap_id / long_press 起作用
            #    （back/swipe/wait 这种导航动作允许合法连击）
            if action.get("action") in ("tap", "tap_id", "long_press"):
                sig = self._action_signature(action)
                recent_actions.append(sig)
                recent_actions = recent_actions[-self.loop_threshold:]
                if (len(recent_actions) >= self.loop_threshold
                        and len(set(recent_actions)) == 1):
                    result.stop_reason = (
                        f"detected loop: 同一坐标连续 tap {self.loop_threshold} 次 ({sig})"
                    )
                    self.log(f"=== Stop: {result.stop_reason} ===")
                    self.log("    提示：模型反复点同一处但页面不前进，"
                             "说明 grounding 错了或目标不在当前屏。看 data/screens/ 对照。")
                    return result
            else:
                recent_actions.clear()  # 非 tap 动作打断了 tap 重复链

            time.sleep(0.4)

        result.stop_reason = (
            f"reached max_steps={self.max_steps}（动作步数上限，不含 wait）"
        )
        self.log(f"=== Stop: {result.stop_reason} ===")
        return result

    # ------------------------------------------------------------------
    def _resize_png(self, raw: bytes) -> Tuple[bytes, Tuple[int, int], float]:
        """缩到 ≤ screenshot_max_width 宽。
        返回 (new_png, (w,h), scale)。scale = device_pixel / image_pixel。"""
        try:
            from PIL import Image
        except ImportError:
            dev_w, dev_h = self.executor.width, self.executor.height
            return raw, (dev_w, dev_h), 1.0

        img = Image.open(io.BytesIO(raw))
        ow, oh = img.size
        if ow <= self.screenshot_max_width:
            new_img = img
            new_w, new_h = ow, oh
        else:
            ratio = self.screenshot_max_width / ow
            new_w = self.screenshot_max_width
            new_h = int(oh * ratio)
            new_img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        new_img.save(buf, format="PNG")
        dev_w, _ = self.executor.width, self.executor.height
        scale = dev_w / new_w if new_w else 1.0
        return buf.getvalue(), (new_w, new_h), scale

    # ------------------------------------------------------------------
    def _build_prompt(self, task: str, history: List[str],
                      img_w: int, img_h: int,
                      elements: Optional[List[Any]] = None) -> str:
        hist = "\n".join(history) or "(无)"
        som = ""
        if elements:
            som = "\n" + _SOM_BLOCK.format(elements=self._elements_text(elements)) + "\n"
        return (
            _SYSTEM.format(img_w=img_w, img_h=img_h)
            + som
            + "\n"
            + _USER_TEMPLATE.format(task=task, history=hist)
        )

    # ---------------- Set-of-Marks helpers ----------------
    def _collect_elements(self) -> List[Any]:
        """抓当前屏幕**可点**的无障碍元素，截断到 som_max_marks 个。
        只标可点元素：tap_id 会去点元素中心，标注纯文本会诱导模型点到无效目标。
        screen() 内部走 uiautomator dump；失败/空就返回 []（退回纯视觉坐标）。"""
        try:
            els = self.executor.screen()
        except Exception as e:
            self.log(f"[SoM]     screen() 失败，退回纯视觉: {e}")
            return []
        clickable = [e for e in els if getattr(e, "clickable", False)]
        return clickable[: self.som_max_marks]

    @staticmethod
    def _elements_text(elements: List[Any]) -> str:
        lines = []
        for i, e in enumerate(elements, 1):
            label = (getattr(e, "label", "") or "").strip().replace("\n", " ")[:30]
            kind = "可点" if getattr(e, "clickable", False) else "文本"
            lines.append(f"[{i}] {label or '(无文字)'} · {kind}")
        return "\n".join(lines)

    def _annotate(self, png: bytes, elements: List[Any], scale: float) -> bytes:
        """在截图上给每个元素画红框 + 左上角编号。bounds 是设备像素，
        除以 scale 映射到截图像素。PIL 不可用时原样返回。"""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return png
        try:
            img = Image.open(io.BytesIO(png)).convert("RGB")
        except Exception:
            return png
        draw = ImageDraw.Draw(img)
        for i, e in enumerate(elements, 1):
            (x1, y1), (x2, y2) = e.bounds
            ix1, iy1 = int(x1 / scale), int(y1 / scale)
            ix2, iy2 = int(x2 / scale), int(y2 / scale)
            draw.rectangle([ix1, iy1, ix2, iy2], outline=(255, 0, 0), width=2)
            tag = str(i)
            tw = 7 * len(tag) + 6
            draw.rectangle([ix1, iy1, ix1 + tw, iy1 + 16], fill=(255, 0, 0))
            draw.text((ix1 + 3, iy1 + 2), tag, fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _element_by_id(self, eid: Any) -> Optional[Any]:
        try:
            i = int(eid)
        except (TypeError, ValueError):
            return None
        els = self._cur_elements
        if 1 <= i <= len(els):
            return els[i - 1]
        return None

    @staticmethod
    def _parse(rsp: str) -> Optional[Dict]:
        t = (rsp or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*", "", t).rstrip("`").strip()

        # 首选：从第一个 '{' 起用 raw_decode 抓"第一个完整 JSON 对象"，
        # 自动忽略对象后面的多余文字（模型常在合法 JSON 后再吐一段思考，
        # 那段里若含 '}' 会让贪婪正则 \{.*\} 抓过头 → 误判非法）。
        start = t.find("{")
        if start != -1:
            decoder = json.JSONDecoder()
            tail = t[start:]
            for cand in (tail, GUIAgent._repair_json(tail)):
                try:
                    obj, _ = decoder.raw_decode(cand)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass

        # 兜底：旧的整体/贪婪匹配策略
        candidates = [t]
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for c in candidates:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(GUIAgent._repair_json(c))
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _repair_json(s: str) -> str:
        """容错修复 VLM 输出里的常见 JSON 小毛病：
           - 数字值后多余引号： "y": 1105"     ->  "y": 1105
           - 漏掉键名：        "x": 405, 1107  ->  "x": 405, "y": 1107
           - 数字后多余方括号： "x": 499, 87]   ->  "x": 499, "y": 87
           - 末尾逗号：        , }             ->  }

        注意第 1 条**必须**锚定在冒号后（值位置），否则会误伤"结尾是数字
        的字符串值"——比如 thought 文本 "...设为8" 的收尾引号会被当成多余引号
        删掉，反而把合法 JSON 改坏（Qwen 常输出以数字结尾的中文 thought）。
        agent 的决策 JSON 里不含数组，所以"数字后紧跟 ] 再接 ,/}"必是杂散括号。
        """
        s = re.sub(r'(:\s*-?\d+(?:\.\d+)?)\s*"(\s*[,}\]])', r'\1\2', s)
        if '"y"' not in s:
            s = re.sub(
                r'("x"\s*:\s*-?\d+(?:\.\d+)?\s*,\s*)(-?\d+(?:\.\d+)?)',
                r'\1"y": \2', s,
            )
        # 数字后多出来的杂散方括号（Qwen 把坐标写成 "x":499, 87] 这种）
        s = re.sub(r'(\d)\s*\]\s*([,}])', r'\1\2', s)
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        return s

    @staticmethod
    def _normalize_action(decision: Dict) -> Dict[str, Any]:
        action: Dict[str, Any] = {}
        a = decision.get("action") or decision.get("action_type")
        if isinstance(a, dict):
            action.update(a)
            a = a.get("type") or a.get("action") or a.get("action_type")
        # 归一动作同义词：不同 VLM 用词不同（Qwen 爱用 click，有的用 input/scroll）
        if isinstance(a, str):
            a = _ACTION_ALIASES.get(a.strip().lower(), a.strip().lower())
        action["action"] = a
        for k in ("x", "y", "id", "text", "direction", "package", "clear"):
            if k in decision and decision[k] is not None:
                action[k] = decision[k]
        return action

    # ------------------------------------------------------------------
    def _dispatch(self, action: Dict[str, Any], scale: float
                  ) -> Tuple[bool, str]:
        a = action.get("action")
        try:
            if a == "tap_id":
                el = self._element_by_id(action.get("id"))
                if el is None:
                    return False, f"tap_id 无效 id={action.get('id')!r}"
                label = getattr(el, "label", "")
                if _is_risky_label(label):
                    self.log(f"[Safety]  风险动作：tap_id 命中 {label!r}")
                    if self.block_risky_actions:
                        return False, f"已拦截风险动作 tap_id {label!r}（block_risky_actions=on）"
                cx, cy = el.center  # 设备像素
                x = max(0, min(int(cx), self.executor.width - 1))
                y = max(0, min(int(cy), self.executor.height - 1))
                self.executor.tap(x, y)
                action["_device_xy"] = (x, y)
                return True, f"tap_id {action.get('id')} -> ({x},{y})"
            if a == "tap":
                x, y = self._to_device_px(action, scale)
                if x is None:
                    return False, "tap 缺少 x/y"
                self.executor.tap(x, y)
                action["_device_xy"] = (x, y)
                return True, f"tap ({x},{y})"
            if a == "long_press":
                x, y = self._to_device_px(action, scale)
                if x is None:
                    return False, "long_press 缺少 x/y"
                self.executor.long_press(x, y)
                action["_device_xy"] = (x, y)
                return True, f"long_press ({x},{y})"
            if a == "type":
                text = action.get("text", "")
                if not text:
                    return False, "type 缺少 text"
                if action.get("clear"):
                    self.executor.clear_text()
                self.executor.type_text(text)
                return True, f"type {text!r}"
            if a == "clear":
                self.executor.clear_text()
                return True, "clear"
            if a == "enter":
                self.executor.enter()
                return True, "enter"
            if a == "swipe":
                direction = action.get("direction", "up")
                ok = self.executor.swipe(direction)
                return ok, f"swipe {direction}"
            if a == "back":
                self.executor.back()
                return True, "back"
            if a == "wait":
                time.sleep(1.0)
                return True, "wait"
            if a == "open_app":
                pkg = action.get("package")
                if not pkg:
                    return False, "open_app 缺少 package"
                self.executor.open_app(pkg)
                return True, f"open {pkg}"
            return False, f"unknown action: {a}"
        except Exception as e:
            return False, f"dispatch error: {e}"

    def _to_device_px(self, action: Dict, scale: float = 1.0
                      ) -> Tuple[Optional[int], Optional[int]]:
        """模型坐标 → 设备像素坐标。

        约定：模型输出 [0, 1000] 归一化坐标（Qwen2-VL/3-VL 标准）。
        若任一坐标 > 1000，则认为模型输出的是**缩放后截图**的绝对像素
        （老模型如 qwen-vl-max），需乘以 scale(=设备像素/截图像素) 还原到设备。
        """
        x = action.get("x")
        y = action.get("y")
        if x is None or y is None:
            return None, None
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            return None, None
        dev_w, dev_h = self.executor.width, self.executor.height
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            px = int(round(x * dev_w / 1000.0))
            py = int(round(y * dev_h / 1000.0))
        else:
            # 绝对像素：模型看到的是缩放后的截图，按 scale 还原到设备像素
            px, py = int(round(x * scale)), int(round(y * scale))
        px = max(0, min(px, dev_w - 1))
        py = max(0, min(py, dev_h - 1))
        return px, py

    @staticmethod
    def _frame_hash(png: bytes) -> Optional[str]:
        """把截图压成 16x16 灰度、量化后取哈希，作为"屏幕是否变化"的粗签名。
        量化(>>4)是为了容忍状态栏时钟/光标闪烁这类细微差异。
        没装 PIL 时返回 None（退化为不做帧检测）。"""
        try:
            from PIL import Image
        except ImportError:
            return None
        import hashlib
        try:
            img = Image.open(io.BytesIO(png)).convert("L").resize((16, 16))
        except Exception:
            return None
        data = bytes((p >> 4) for p in img.getdata())
        return hashlib.md5(data).hexdigest()

    @staticmethod
    def _action_signature(action: Dict) -> str:
        """循环检测用的动作签名：动作类型 + 关键参数。"""
        a = action.get("action", "?")
        bits = [a]
        if "x" in action and "y" in action:
            bits.append(f"{action['x']},{action['y']}")
        for k in ("id", "text", "direction", "package"):
            if action.get(k) is not None:
                bits.append(f"{k}={action[k]}")
        return "|".join(bits)

    @staticmethod
    def _summarize_step(action: Dict, ok: bool) -> str:
        a = action.get("action")
        bits = [a or "?"]
        if action.get("id") is not None:
            bits.append(f"#{action['id']}")
        if "x" in action and "y" in action:
            bits.append(f"({action['x']},{action['y']})")
        if action.get("text"):
            bits.append(f"text={action['text']!r}")
        if action.get("direction"):
            bits.append(action["direction"])
        if action.get("package"):
            bits.append(action["package"])
        bits.append("OK" if ok else "FAIL")
        return " ".join(bits)
