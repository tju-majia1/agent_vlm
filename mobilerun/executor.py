"""
AdbExecutor —— 真机执行器

只用 Python 标准库 + adb 命令行，不需要在手机上装任何额外控制 APK。

约束：
- adb 必须在 PATH 里
- 手机连 USB 或同网段 adb tcpip 连接，开发者模式开 USB 调试
- adb shell input text 默认只支持 ASCII；中文输入需在手机装 ADBKeyBoard
  （详见 README 的"真机演示"小节）

提供两个核心方法以满足 SkillRunner 的 DeviceExecutor 协议：
    screen() -> List[ScreenElement]   # 解析当前 UI 树
    act(step: SkillStep) -> bool      # 执行 click / type / swipe / back / open_app
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

from mobilerun.schema import SkillStep
from mobilerun.self_heal import ScreenElement


class AdbError(RuntimeError):
    pass


def _find_adb() -> str:
    """定位 adb 可执行文件，按优先级：
       1) 环境变量 ADB_PATH（显式指定）
       2) 已在 PATH 里
       3) 常见 Android SDK 安装位置（ANDROID_HOME / LOCALAPPDATA 等）
       都找不到就退回字面量 "adb"，让后续报清晰错误。
    这样即便用户没把 platform-tools 加进 PATH 也能直接跑。"""
    p = os.environ.get("ADB_PATH")
    if p and os.path.isfile(p):
        return p
    w = shutil.which("adb")
    if w:
        return w
    exe = "adb.exe" if os.name == "nt" else "adb"
    bases = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Android", "Sdk"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk"),
    ]
    for base in bases:
        if not base:
            continue
        cand = os.path.join(base, "platform-tools", exe)
        if os.path.isfile(cand):
            return cand
    return "adb"


# 解析一次，缓存供全模块复用
_ADB = _find_adb()


# ----------------------------------------------------------------------
# AdbExecutor
# ----------------------------------------------------------------------
class AdbExecutor:
    def __init__(self, serial: str = "", *, action_pause: float = 0.6,
                 wait_after_open: float = 2.0):
        """
        Args:
            serial: 指定设备 id（adb devices 出来的那个）。空表示用第一个设备。
            action_pause: 每个动作执行后暂停（让 UI 反应）
            wait_after_open: open_app 后多等一会儿等首屏起来
        """
        self.serial = serial or self._auto_pick_serial()
        self.action_pause = action_pause
        self.wait_after_open = wait_after_open
        self.width, self.height = self._device_size()

    # ------------------------------------------------------------------
    # 工厂：批量获取所有连接的设备
    # ------------------------------------------------------------------
    @staticmethod
    def list_devices() -> List[str]:
        out = _adb_raw(["devices"])
        devs = []
        for line in out.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devs.append(parts[0])
        return devs

    def _auto_pick_serial(self) -> str:
        devs = self.list_devices()
        if not devs:
            raise AdbError("没找到任何 adb 设备。请插 USB / 开 USB 调试 / 信任本电脑。")
        return devs[0]

    # ------------------------------------------------------------------
    # 设备元信息
    # ------------------------------------------------------------------
    def _adb(self, *args: str, timeout: float = 30.0) -> str:
        return _adb_raw(["-s", self.serial, *args], timeout=timeout)

    def _device_size(self) -> Tuple[int, int]:
        out = self._adb("shell", "wm", "size")
        m = re.search(r"(\d+)x(\d+)", out)
        if not m:
            raise AdbError(f"无法解析屏幕尺寸: {out!r}")
        return int(m.group(1)), int(m.group(2))

    # ------------------------------------------------------------------
    # screenshot()：抓屏幕图（PNG bytes + 设备分辨率）
    # 这是 vision-grounded agent 的主要观察接口
    # ------------------------------------------------------------------
    def screenshot(self) -> bytes:
        """直接拉取一张 PNG。adb exec-out 比 shell screencap 然后 pull 快很多。
        对超时/空图/损坏（非 PNG 头）做几次重试，扛住偶发抖动。"""
        last: Optional[AdbError] = None
        for attempt in range(3):
            try:
                r = subprocess.run(
                    [_ADB, "-s", self.serial, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=15.0,
                )
            except subprocess.TimeoutExpired:
                last = AdbError("截图超时")
            else:
                if r.returncode == 0 and r.stdout[:8] == b"\x89PNG\r\n\x1a\n":
                    return r.stdout
                last = AdbError(
                    "截图失败: "
                    + (r.stderr.decode(errors="ignore").strip()[:120] or "空/非PNG数据")
                )
            if attempt < 2:
                time.sleep(0.5)
        raise last if last else AdbError("截图失败")

    def tap(self, x: int, y: int):
        """直接按坐标点击（vision agent 用）"""
        self._adb("shell", "input", "tap", str(x), str(y))
        self._pause()

    def type_text(self, text: str):
        """直接输入文字。中文走 ADBKeyBoard，ASCII 走原生。"""
        if _is_ascii(text):
            self._adb("shell", "input", "text", _escape_for_input_text(text))
        else:
            ok = self._type_via_adbkeyboard(text)
            if not ok:
                self._adb("shell", "input", "text", _escape_for_input_text(text))
        self._pause()

    def back(self):
        self._adb("shell", "input", "keyevent", "KEYCODE_BACK")
        self._pause()

    def enter(self):
        """回车 / 提交（搜索框输完字直接发起搜索）。"""
        self._adb("shell", "input", "keyevent", "66")  # KEYCODE_ENTER
        self._pause()

    def clear_text(self):
        """清空当前焦点输入框。优先 ADBKeyBoard 的 ADB_CLEAR_TEXT 广播，
        回退到 移到末尾 + 连续删除（兜底，慢但通用）。"""
        try:
            self._adb("shell", "am", "broadcast", "-a", "ADB_CLEAR_TEXT")
            self._pause()
            return
        except AdbError:
            pass
        # 一条命令里连发：移到末尾(123) + 多次删除(67)，避免几十次进程往返
        self._adb("shell", "input", "keyevent", "123", *(["67"] * 60))
        self._pause()

    def long_press(self, x: int, y: int, duration_ms: int = 1000):
        """用同点 swipe 制造长按（vision agent 用）。"""
        self._adb("shell", "input", "swipe",
                  str(x), str(y), str(x), str(y), str(duration_ms))
        self._pause()

    def swipe(self, direction: str) -> bool:
        """方向滑动（vision agent 用）。"""
        return self._do_swipe(SkillStep(action_type="swipe", direction=direction))

    def open_app(self, package: str):
        """优先解析出 LAUNCHER activity 再 am start（干净、可控）；
        解析不到就回退 monkey（噪声大但通用）。"""
        comp = self._resolve_launch_component(package)
        try:
            if comp:
                self._adb("shell", "am", "start", "-n", comp)
            else:
                self._adb("shell", "monkey", "-p", package,
                          "-c", "android.intent.category.LAUNCHER", "1")
        except AdbError:
            self._adb("shell", "monkey", "-p", package,
                      "-c", "android.intent.category.LAUNCHER", "1")
        time.sleep(self.wait_after_open)

    def _resolve_launch_component(self, package: str) -> Optional[str]:
        try:
            out = self._adb(
                "shell", "cmd", "package", "resolve-activity", "--brief",
                "-a", "android.intent.action.MAIN",
                "-c", "android.intent.category.LAUNCHER", package,
            )
        except AdbError:
            return None
        # --brief 末行通常就是 "pkg/.Activity"
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith(package + "/"):
                return line
        return None

    # ------------------------------------------------------------------
    # screen()：抓 UI 树（保留给 skill self-heal 兜底用，agent 主循环不再依赖）
    # ------------------------------------------------------------------
    def screen(self) -> List[ScreenElement]:
        xml = self._read_uiautomator_dump()
        if not xml:
            return []
        return _parse_ui_xml(xml)

    def _read_uiautomator_dump(self) -> Optional[str]:
        """dump UI 树到 /sdcard 后用 exec-out cat 直接读进内存。
        比旧的 `adb pull` 到临时文件再读少一次往返、无磁盘 IO。"""
        device_path = "/sdcard/uia_dump.xml"
        try:
            self._adb("shell", "uiautomator", "dump", device_path)
        except AdbError:
            return None
        try:
            r = subprocess.run(
                [_ADB, "-s", self.serial, "exec-out", "cat", device_path],
                capture_output=True, timeout=20.0,
            )
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not r.stdout:
            return None
        return r.stdout.decode("utf-8", "ignore")

    # ------------------------------------------------------------------
    # act()：执行动作
    # ------------------------------------------------------------------
    def act(self, step: SkillStep) -> bool:
        a = step.action_type
        try:
            if a == "click":
                return self._do_click(step)
            if a == "long_press":
                return self._do_long_press(step)
            if a == "type":
                return self._do_type(step)
            if a == "swipe":
                return self._do_swipe(step)
            if a == "back":
                self._adb("shell", "input", "keyevent", "KEYCODE_BACK")
                self._pause()
                return True
            if a == "wait":
                time.sleep(1.0)
                return True
            if a == "open_app":
                return self._do_open_app(step)
            return False
        except AdbError:
            return False

    # ------------------------------------------------------------------
    # 具体动作
    # ------------------------------------------------------------------
    def _do_click(self, step: SkillStep) -> bool:
        x, y = self._resolve_xy(step)
        if x is None:
            return False
        self._adb("shell", "input", "tap", str(x), str(y))
        self._pause()
        return True

    def _do_long_press(self, step: SkillStep) -> bool:
        x, y = self._resolve_xy(step)
        if x is None:
            return False
        # 用 swipe 同点制造长按
        self._adb("shell", "input", "swipe",
                  str(x), str(y), str(x), str(y), "1000")
        self._pause()
        return True

    def _do_type(self, step: SkillStep) -> bool:
        text = step.text or ""
        if not text:
            return False
        # 如果指定了目标控件，先点一下让焦点进去
        if step.target_text or step.coordinates:
            x, y = self._resolve_xy(step)
            if x is not None:
                self._adb("shell", "input", "tap", str(x), str(y))
                time.sleep(0.3)
        # 用 ADBKeyBoard 输中文（如果装了），否则走原生 input text（ASCII）
        if _is_ascii(text):
            self._adb("shell", "input", "text", _escape_for_input_text(text))
        else:
            ok = self._type_via_adbkeyboard(text)
            if not ok:
                # 兜底：警告并尝试直接发（多数 ROM 不支持中文，会得到乱码或空）
                self._adb("shell", "input", "text", _escape_for_input_text(text))
        self._pause()
        return True

    def _type_via_adbkeyboard(self, text: str) -> bool:
        # 通过广播给 ADBKeyBoard，它会把文字注入到当前焦点
        # 用 base64 通道（ADB_INPUT_B64）绕开所有 shell 引号/特殊字符问题
        # APK: https://github.com/senzhk/ADBKeyBoard
        import base64
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        try:
            self._adb(
                "shell",
                "am", "broadcast",
                "-a", "ADB_INPUT_B64",
                "--es", "msg", b64,
            )
            return True
        except AdbError:
            return False

    def _do_swipe(self, step: SkillStep) -> bool:
        """大幅度滑动：竖直走屏高的 ~55%、水平走屏宽的 ~55%。
        短滑（旧的 1/4 屏、居中起手）在 Pixel 桌面打不开应用抽屉、
        长列表也滚不动，所以加大行程并把起点放在更靠边的位置。"""
        direction = step.direction or "up"
        w, h = self.width, self.height
        cx, cy = w // 2, h // 2
        if direction == "up":          # 内容上移 / 打开应用抽屉
            sx, sy, ex, ey = cx, int(h * 0.78), cx, int(h * 0.22)
        elif direction == "down":      # 内容下移
            sx, sy, ex, ey = cx, int(h * 0.22), cx, int(h * 0.78)
        elif direction == "left":
            sx, sy, ex, ey = int(w * 0.82), cy, int(w * 0.18), cy
        elif direction == "right":
            sx, sy, ex, ey = int(w * 0.18), cy, int(w * 0.82), cy
        else:
            return False
        self._adb("shell", "input", "swipe",
                  str(sx), str(sy), str(ex), str(ey), "350")
        self._pause()
        return True

    def _do_open_app(self, step: SkillStep) -> bool:
        pkg = step.package or step.target_text
        if not pkg:
            return False
        try:
            self._adb("shell", "monkey", "-p", pkg,
                      "-c", "android.intent.category.LAUNCHER", "1")
        except AdbError:
            return False
        time.sleep(self.wait_after_open)
        return True

    # ------------------------------------------------------------------
    # 坐标解析：优先用 step.coordinates；没坐标就靠 target_text 在 UI 树里查
    # ------------------------------------------------------------------
    def _resolve_xy(self, step: SkillStep) -> Tuple[Optional[int], Optional[int]]:
        if step.coordinates and len(step.coordinates) == 2:
            return step.coordinates[0], step.coordinates[1]
        target = (step.target_text or "").strip()
        if not target:
            return None, None
        for e in self.screen():
            if e.text == target or e.content_desc == target:
                if e.clickable:
                    return e.center
        # 退化：包含匹配
        for e in self.screen():
            if target and (target in e.text or target in e.content_desc):
                return e.center
        return None, None

    def _pause(self):
        time.sleep(self.action_pause)


# ----------------------------------------------------------------------
# adb 进程封装
# ----------------------------------------------------------------------
# 这些 stderr 关键词通常是瞬时的（设备短暂离线/守护进程抖动），值得重试
_TRANSIENT_ADB = (
    "device offline", "device still authorizing", "closed", "protocol fault",
    "no devices", "device not found", "device unauthorized",
    "daemon not running", "cannot connect", "connection reset",
)


def _adb_raw(args: List[str], timeout: float = 30.0, retries: int = 2) -> str:
    """执行一条 adb 命令。对超时 / 设备瞬时离线等可恢复错误做几次退避重试，
    避免真机/模拟器偶发一次抖动就把整个任务打断。非瞬时失败立即抛出。"""
    last: Optional[AdbError] = None
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(
                [_ADB, *args],
                capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError as e:
            raise AdbError(
                f"找不到 adb 命令（尝试用 {_ADB!r}）。请装 platform-tools 并加入 PATH，"
                "或设环境变量 ADB_PATH 指向 adb.exe。"
            ) from e
        except subprocess.TimeoutExpired:
            last = AdbError(f"adb 命令超时: {' '.join(args)}")
        else:
            if r.returncode == 0:
                return r.stdout.strip()
            stderr = (r.stderr or "").strip()
            if (attempt < retries
                    and any(m in stderr.lower() for m in _TRANSIENT_ADB)):
                last = AdbError(f"adb 瞬时错误: {stderr}")
            else:
                raise AdbError(
                    f"adb 失败 (returncode={r.returncode}): "
                    f"args={' '.join(args)}, stderr={stderr}"
                )
        if attempt < retries:
            time.sleep(0.6 * (attempt + 1))
    raise last if last else AdbError(f"adb 失败: {' '.join(args)}")


# ----------------------------------------------------------------------
# UI 树解析
# ----------------------------------------------------------------------
def _parse_ui_xml(xml_text: str) -> List[ScreenElement]:
    elems: List[ScreenElement] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return elems

    seen_centers: List[Tuple[int, int]] = []
    for node in root.iter("node"):
        a = node.attrib
        clickable = a.get("clickable") == "true"
        text = (a.get("text") or "").strip()
        desc = (a.get("content-desc") or "").strip()
        if not (clickable or text or desc):
            continue
        bounds = _parse_bounds(a.get("bounds", ""))
        if not bounds:
            continue
        (x1, y1), (x2, y2) = bounds
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # 去重：中心点距 < 20px 的视为同一控件
        close = False
        for sx, sy in seen_centers:
            if (cx - sx) ** 2 + (cy - sy) ** 2 < 400:
                close = True
                break
        if close:
            continue
        seen_centers.append((cx, cy))
        elems.append(ScreenElement(
            text=text, content_desc=desc,
            bounds=bounds, clickable=clickable,
        ))
    return elems


def _parse_bounds(s: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    # "[x1,y1][x2,y2]"
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2))), (int(m.group(3)), int(m.group(4)))


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# adb `input text` 把 %s 当空格；其余 shell 元字符需反斜杠转义，
# 否则 `'`、`&`、`(`、`<` 等会被设备端 shell 吃掉或截断输入。
_INPUT_SPECIAL = set("()<>|;&*\\~\"'`$# ")


def _escape_for_input_text(text: str) -> str:
    out = []
    for ch in text:
        if ch == " ":
            out.append("%s")
        elif ch in _INPUT_SPECIAL:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)
