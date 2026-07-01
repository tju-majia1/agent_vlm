"""
AdbExecutor 纯函数 / 解析逻辑单测（不连真机）。

跑法：
    pytest tests/test_executor_core.py
"""

from __future__ import annotations

from mobilerun.executor import (
    _escape_for_input_text, _is_ascii, _parse_bounds, _parse_ui_xml,
    _find_adb, _TRANSIENT_ADB,
)


# ---------------------------------------------------------------- input 转义
def test_escape_space_to_percent_s():
    assert _escape_for_input_text("a b c") == "a%sb%sc"


def test_escape_shell_specials():
    assert _escape_for_input_text("a'b") == r"a\'b"
    assert _escape_for_input_text("x&y(z)") == r"x\&y\(z\)"


def test_escape_plain_untouched():
    assert _escape_for_input_text("Hello123") == "Hello123"


def test_escape_non_ascii_passthrough():
    # 中文字符不在转义集里，原样保留（实际会走 ADBKeyBoard，不走这里）
    assert _escape_for_input_text("北京") == "北京"


# ---------------------------------------------------------------- _is_ascii
def test_is_ascii():
    assert _is_ascii("hello-123_!")
    assert not _is_ascii("你好")
    assert not _is_ascii("café")


# ---------------------------------------------------------------- bounds 解析
def test_parse_bounds_ok():
    assert _parse_bounds("[0,10][100,210]") == ((0, 10), (100, 210))


def test_parse_bounds_bad():
    assert _parse_bounds("garbage") is None
    assert _parse_bounds("") is None


# ---------------------------------------------------------------- UI XML 解析（字符串）
def test_parse_ui_xml_from_string():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<hierarchy>'
        '<node clickable="true" text="OK" content-desc="" bounds="[0,0][200,100]"/>'
        '<node clickable="false" text="Label" content-desc="" bounds="[0,200][200,300]"/>'
        '<node clickable="true" text="" content-desc="" bounds="[0,0][1,1]"/>'  # 太小，丢弃
        '</hierarchy>'
    )
    els = _parse_ui_xml(xml)
    labels = {e.label for e in els}
    assert "OK" in labels and "Label" in labels
    ok = [e for e in els if e.label == "OK"][0]
    assert ok.clickable and ok.center == (100, 50)


def test_parse_ui_xml_bad_returns_empty():
    assert _parse_ui_xml("not xml at all <") == []


# ---------------------------------------------------------------- _find_adb
def test_find_adb_prefers_adb_path_env(monkeypatch, tmp_path):
    fake = tmp_path / "adb.exe"
    fake.write_text("x")
    monkeypatch.setenv("ADB_PATH", str(fake))
    assert _find_adb() == str(fake)


def test_transient_markers_present():
    # 守住几个关键的瞬时错误标记，别被误删
    for m in ("device offline", "daemon not running", "device not found"):
        assert m in _TRANSIENT_ADB
