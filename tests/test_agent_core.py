"""
GUIAgent 核心纯函数单元测试（不连真机、不调真 LLM）。

覆盖之前完全没测到的主路径逻辑：JSON 容错 / 解析 / 动作归一化 /
归一化坐标换算 / 帧哈希 / 成功轨迹蒸馏。

跑法：
    pytest tests/test_agent_core.py
"""

from __future__ import annotations

import io

from mobilerun.agent import GUIAgent, GUIAgentResult, GUIAgentStep


class _DummyExecutor:
    """只提供 _to_device_px 需要的设备分辨率，并记录动作。"""
    width = 1080
    height = 2400

    def __init__(self):
        self.taps = []
        self.events = []

    def tap(self, x, y):
        self.taps.append((x, y))
        self.events.append(("tap", x, y))

    def enter(self):
        self.events.append(("enter",))

    def clear_text(self):
        self.events.append(("clear",))

    def type_text(self, text):
        self.events.append(("type", text))


class _DummyLLM:
    def chat(self, prompt, *, image=None):
        return "{}"


class _ScriptLLM:
    """按脚本依次返回决策（最后一个会重复）。"""
    def __init__(self, decisions):
        self.decisions = decisions
        self.i = 0

    def chat(self, prompt, *, image=None):
        import json
        d = self.decisions[min(self.i, len(self.decisions) - 1)]
        self.i += 1
        return json.dumps(d, ensure_ascii=False)


def _png_bytes(color=(255, 255, 255)):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(b, format="PNG")
    return b.getvalue()


class _Elem:
    """模拟 ScreenElement（self_heal.ScreenElement 的最小替身）。"""
    def __init__(self, label, bounds, clickable=True):
        self._label = label
        self.bounds = bounds
        self.clickable = clickable

    @property
    def label(self):
        return self._label

    @property
    def center(self):
        (x1, y1), (x2, y2) = self.bounds
        return (x1 + x2) // 2, (y1 + y2) // 2


def _agent() -> GUIAgent:
    return GUIAgent(_DummyExecutor(), _DummyLLM(), screens_dir=None)


# ---------------------------------------------------------------- _repair_json
def test_repair_trailing_quote_on_number():
    assert GUIAgent._repair_json('{"y": 1105"}') == '{"y": 1105}'


def test_repair_missing_y_key():
    out = GUIAgent._repair_json('{"x": 405, 1107}')
    assert '"y": 1107' in out


def test_repair_trailing_comma():
    assert GUIAgent._repair_json('{"a": 1,}') == '{"a": 1}'


# ---------------------------------------------------------------- _parse
def test_parse_code_fence():
    d = GUIAgent._parse('```json\n{"action":"tap","x":1,"y":2}\n```')
    assert d == {"action": "tap", "x": 1, "y": 2}


def test_parse_embedded_json():
    d = GUIAgent._parse('好的，下一步：{"action":"back"} 就这样')
    assert d == {"action": "back"}


def test_parse_repairs_then_loads():
    d = GUIAgent._parse('{"x": 405, 1107}')
    assert d == {"x": 405, "y": 1107}


def test_parse_returns_none_on_garbage():
    assert GUIAgent._parse("完全不是 json") is None


def test_parse_json_followed_by_prose_with_braces():
    # 真实失败样本：合法 JSON 后跟一段思考，思考里含 '}'，
    # 贪婪正则会抓过头；raw_decode 必须只取第一个对象。
    rsp = ('{"action": "tap", "x": 101, "y": 771, "done": false}\n\n'
           'Wait, let me reconsider {this} based on dims (1080x2424).')
    d = GUIAgent._parse(rsp)
    assert d == {"action": "tap", "x": 101, "y": 771, "done": False}


def test_parse_json_with_leading_prose():
    d = GUIAgent._parse('Looking at the screen: {"action":"back","done":false} done.')
    assert d == {"action": "back", "done": False}


def test_repair_does_not_eat_string_ending_in_digit():
    # 真实失败样本（Qwen）：漏了 "y" 键，且 thought 以数字结尾。
    # 修复 y 时绝不能把 "设为8" 的收尾引号删掉。
    rsp = '{"action": "tap", "x": 294, 623, "thought": "数字8设为8", "done": false}'
    d = GUIAgent._parse(rsp)
    assert d["action"] == "tap"
    assert d["x"] == 294 and d["y"] == 623
    assert d["thought"] == "数字8设为8"  # thought 完整保留


def test_repair_still_strips_spurious_quote_after_number_value():
    assert GUIAgent._repair_json('{"y": 1105"}') == '{"y": 1105}'


def test_parse_missing_y_with_stray_bracket():
    # 真实失败样本（Qwen）：漏 "y" 键 + 数字后多了个杂散 ']'
    rsp = '{"action": "tap", "x": 499, 87], "thought": "点击搜索栏", "done": false}'
    d = GUIAgent._parse(rsp)
    assert d["action"] == "tap" and d["x"] == 499 and d["y"] == 87


# ---------------------------------------------------------------- _normalize_action
def test_normalize_flat():
    a = GUIAgent._normalize_action(
        {"action": "tap", "x": 1, "y": 2, "thought": "x"})
    assert a["action"] == "tap" and a["x"] == 1 and a["y"] == 2


def test_normalize_nested_action_dict():
    a = GUIAgent._normalize_action(
        {"action": {"type": "swipe", "direction": "up"}})
    assert a["action"] == "swipe" and a["direction"] == "up"


def test_normalize_action_type_alias():
    a = GUIAgent._normalize_action({"action_type": "back"})
    assert a["action"] == "back"


def test_normalize_action_synonyms():
    # 不同 VLM 的用词都要归一到本 agent 的词汇
    assert GUIAgent._normalize_action({"action": "click", "x": 1, "y": 2})["action"] == "tap"
    assert GUIAgent._normalize_action({"action": "Click"})["action"] == "tap"
    assert GUIAgent._normalize_action({"action": "scroll", "direction": "up"})["action"] == "swipe"
    assert GUIAgent._normalize_action({"action": "input", "text": "x"})["action"] == "type"
    assert GUIAgent._normalize_action({"action": "go_back"})["action"] == "back"
    assert GUIAgent._normalize_action({"action": "done"})["action"] == "finish"
    # 已是规范词的不受影响
    assert GUIAgent._normalize_action({"action": "tap_id", "id": 1})["action"] == "tap_id"


# ---------------------------------------------------------------- _to_device_px
def test_to_device_px_normalized_center():
    ag = _agent()
    px, py = ag._to_device_px({"x": 500, "y": 500})
    assert px == 540 and py == 1200


def test_to_device_px_normalized_corner_clamped():
    ag = _agent()
    px, py = ag._to_device_px({"x": 1000, "y": 1000})
    # 1000 -> 1080/2400，但要 clamp 到 w-1/h-1
    assert px == 1079 and py == 2399


def test_to_device_px_absolute_uses_scale():
    ag = _agent()
    # 截图被缩到一半（scale=2.0 表示 设备像素=截图像素*2），坐标 > 1000 视为绝对像素
    px, py = ag._to_device_px({"x": 1100, "y": 1100}, scale=2.0)
    assert px == 1079  # 2200 clamp 到 1079
    assert py == 2200


def test_to_device_px_missing_returns_none():
    ag = _agent()
    assert ag._to_device_px({"x": 5}) == (None, None)


# ---------------------------------------------------------------- _frame_hash
def test_frame_hash_stable_and_discriminating():
    try:
        from PIL import Image
    except ImportError:
        return  # 没装 PIL 就跳过

    def _png(color):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buf, format="PNG")
        return buf.getvalue()

    h_white1 = GUIAgent._frame_hash(_png((255, 255, 255)))
    h_white2 = GUIAgent._frame_hash(_png((255, 255, 255)))
    h_black = GUIAgent._frame_hash(_png((0, 0, 0)))
    assert h_white1 is not None
    assert h_white1 == h_white2          # 相同画面 → 相同哈希
    assert h_white1 != h_black           # 不同画面 → 不同哈希


# ---------------------------------------------------------------- trace_for_skill
def test_trace_maps_tap_to_click_and_uses_device_xy():
    res = GUIAgentResult(task="t", success=True)
    res.steps.append(GUIAgentStep(
        1, "", {"action": "tap", "x": 500, "y": 500,
                "_device_xy": (540, 1200)}, True))
    res.steps.append(GUIAgentStep(
        2, "", {"action": "type", "text": "hi"}, True))
    res.steps.append(GUIAgentStep(
        3, "", {"action": "finish"}, True))  # finish 应被剔除
    res.steps.append(GUIAgentStep(
        4, "", {"action": "tap", "x": 1, "y": 1}, False))  # 失败步应被剔除

    trace = res.trace_for_skill()
    assert len(trace) == 2
    assert trace[0]["action_type"] == "click"
    assert trace[0]["coordinates"] == [540, 1200]  # 用设备像素而非归一化值
    assert trace[1] == {"action_type": "type", "text": "hi"}


# ---------------------------------------------------------------- Set-of-Marks
def test_normalize_keeps_id():
    a = GUIAgent._normalize_action({"action": "tap_id", "id": 3, "thought": "x"})
    assert a["action"] == "tap_id" and a["id"] == 3


def test_elements_text_numbers_from_one():
    els = [_Elem("AM", ((800, 600), (900, 700))),
           _Elem("PM", ((800, 700), (900, 800))),
           _Elem("", ((0, 0), (10, 10)), clickable=False)]
    txt = GUIAgent._elements_text(els)
    assert "[1] AM" in txt and "[2] PM" in txt
    assert "[3] (无文字)" in txt


def test_collect_elements_clickable_first():
    ag = _agent()
    ag.executor.screen = lambda: [
        _Elem("label", ((0, 0), (50, 50)), clickable=False),
        _Elem("btn", ((0, 60), (50, 110)), clickable=True),
    ]
    els = ag._collect_elements()
    assert els[0].label == "btn"  # 可点的排前面


def test_collect_elements_empty_on_failure():
    ag = _agent()
    def boom():
        raise RuntimeError("uiautomator down")
    ag.executor.screen = boom
    assert ag._collect_elements() == []  # 失败 → 退回纯视觉


def test_dispatch_tap_id_taps_element_center():
    ag = _agent()
    # AM at y 600-700 (center 650), PM at y 700-800 (center 750)
    ag._cur_elements = [_Elem("AM", ((800, 600), (1000, 700))),
                        _Elem("PM", ((800, 700), (1000, 800)))]
    ok, msg = ag._dispatch({"action": "tap_id", "id": 1}, scale=1.0)
    assert ok
    assert ag.executor.taps[-1] == (900, 650)  # 点到 AM 的中心，不是 PM


def test_dispatch_tap_id_invalid_id():
    ag = _agent()
    ag._cur_elements = [_Elem("only", ((0, 0), (10, 10)))]
    ok, msg = ag._dispatch({"action": "tap_id", "id": 9}, scale=1.0)
    assert not ok and "无效" in msg


def test_build_prompt_includes_som_block_only_when_elements():
    ag = _agent()
    els = [_Elem("Search", ((0, 0), (100, 50)))]
    with_som = ag._build_prompt("t", [], 1080, 2400, els)
    without = ag._build_prompt("t", [], 1080, 2400, None)
    assert "tap_id" in with_som and "[1] Search" in with_som
    assert "tap_id" not in without  # 纯视觉模式 prompt 不变


# ---------------------------------------------------------------- wait budget
def test_wait_does_not_consume_action_budget():
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return
    exe = _DummyExecutor()
    png = _png_bytes()
    exe.screenshot = lambda: png
    # 3 个 wait + finish；max_steps=2。旧逻辑会在 2 个 wait 后耗尽 max_steps，
    # 新逻辑 wait 不计入动作预算 → 应当跑到 finish 成功。
    llm = _ScriptLLM([{"action": "wait"}, {"action": "wait"},
                      {"action": "wait"}, {"action": "finish", "done": True}])
    ag = GUIAgent(exe, llm, max_steps=2, use_som=False, screens_dir=None,
                  wait_base_seconds=0.0, max_consecutive_waits=10, log=lambda *a: None)
    res = ag.run("t")
    assert res.success and res.stop_reason == "done"


def test_consecutive_wait_cap_stops():
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return
    exe = _DummyExecutor()
    png = _png_bytes()
    exe.screenshot = lambda: png
    llm = _ScriptLLM([{"action": "wait"}])  # 永远 wait
    ag = GUIAgent(exe, llm, max_steps=5, use_som=False, screens_dir=None,
                  wait_base_seconds=0.0, max_consecutive_waits=3, log=lambda *a: None)
    res = ag.run("t")
    assert not res.success and "等待超时" in res.stop_reason


# ---------------------------------------------------------------- enter / clear
def test_dispatch_enter():
    ag = _agent()
    ok, _ = ag._dispatch({"action": "enter"}, 1.0)
    assert ok and ("enter",) in ag.executor.events


def test_dispatch_clear():
    ag = _agent()
    ok, _ = ag._dispatch({"action": "clear"}, 1.0)
    assert ok and ("clear",) in ag.executor.events


def test_dispatch_type_with_clear_flag():
    ag = _agent()
    ok, _ = ag._dispatch({"action": "type", "text": "北京", "clear": True}, 1.0)
    assert ok
    assert ("clear",) in ag.executor.events
    assert ("type", "北京") in ag.executor.events
    # clear 必须发生在 type 之前
    assert ag.executor.events.index(("clear",)) < ag.executor.events.index(("type", "北京"))


def test_dispatch_type_without_clear_does_not_clear():
    ag = _agent()
    ag._dispatch({"action": "type", "text": "hi"}, 1.0)
    assert ("clear",) not in ag.executor.events


def test_normalize_enter_clear_aliases():
    assert GUIAgent._normalize_action({"action": "submit"})["action"] == "enter"
    assert GUIAgent._normalize_action({"action": "search"})["action"] == "enter"
    assert GUIAgent._normalize_action({"action": "clear_text"})["action"] == "clear"
    assert GUIAgent._normalize_action({"action": "type", "clear": True})["clear"] is True


# ---------------------------------------------------------------- safety guardrail
def test_risky_guardrail_blocks_when_on():
    ag = GUIAgent(_DummyExecutor(), _DummyLLM(), screens_dir=None,
                  block_risky_actions=True, log=lambda *a: None)
    ag._cur_elements = [_Elem("卸载", ((0, 0), (100, 100)))]
    ok, msg = ag._dispatch({"action": "tap_id", "id": 1}, 1.0)
    assert not ok and "拦截" in msg
    assert ag.executor.taps == []  # 没有真的点下去


def test_risky_guardrail_allows_when_off():
    ag = GUIAgent(_DummyExecutor(), _DummyLLM(), screens_dir=None,
                  block_risky_actions=False, log=lambda *a: None)
    ag._cur_elements = [_Elem("Uninstall", ((0, 0), (100, 100)))]
    ok, _ = ag._dispatch({"action": "tap_id", "id": 1}, 1.0)
    assert ok and ag.executor.taps == [(50, 50)]  # 仅告警，仍执行
