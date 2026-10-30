"""app.py（Streamlit 界面）与 export_graph.py 测试。用 AppTest 无头运行，不调用真实模型。"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import export_graph
import graph as g

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-999"


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def start() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    return at


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    return at


def all_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "warning", "info", "error", "code", "caption", "json", "title", "subheader"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


def test_demo_mode_without_key():
    at = start()
    assert not at.exception
    assert any("演示模式（规则版）" in e.value for e in at.info)
    assert "演示模式（规则版）" in all_text(at)


def test_disclaimer_on_top():
    at = start()
    assert "仅供参考" in at.warning[0].value and "请勿输入真实个人信息" in at.warning[0].value


def test_model_mode_label_does_not_show_key(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    at = start()  # 不提问，因此不会发起模型请求
    text = all_text(at)
    assert "模型模式" in text
    assert FAKE_KEY not in text


def test_chat_turn_and_sidebar():
    at = ask(start(), "成均馆大学需要 GMAT 吗？")
    assert not at.exception
    assert [m.avatar for m in at.chat_message] == ["user", "assistant"]
    assert "选择性提交" in at.chat_message[1].markdown[0].value
    sidebar = "\n".join(str(e.value) for e in [*at.sidebar.code, *at.sidebar.markdown])
    assert "router → requirement_retriever → verify" in sidebar
    assert "https://gsb.skku.edu/gsb/Full-time_MBA/Admissions.do（整理日期 2026-10-29）" in sidebar
    assert "数据整理日期：** 2026-10-29" in sidebar


def test_node_path_filters_internal_entries():
    at = start()
    at.session_state["last_result"] = {"trace": ["router", "router:llm", "requirement_retriever", "verify", "verify:llm_pass"]}
    at.run()
    codes = [c.value for c in at.sidebar.code]
    assert codes == [
        "router → requirement_retriever → verify",
        "router → router:llm → requirement_retriever → verify → verify:llm_pass",
    ]
    expander = at.sidebar.expander[0]
    assert expander.label == "完整 trace"
    assert [c.value for c in expander.code] == [codes[1]]


def test_profile_persists_across_turns():
    at = ask(start(), "我想选校，GPA 3.5")
    assert at.session_state["profile"] == {"gpa": "3.5"}
    ask(at, "选校：本科专业金融，雅思 7.0")
    assert at.session_state["profile"] == {"gpa": "3.5", "major": "金融", "language_score": "雅思 7.0"}
    assert "school_matcher" in at.session_state["last_result"]["trace"]
    assert len(at.chat_message) == 4


def test_question_limit():
    at = start()
    at.session_state["question_count"] = 20
    at.run()
    assert any("20 个问题的上限" in w.value for w in at.warning)
    assert at.chat_input[0].disabled


def test_graph_error_shows_friendly_message(monkeypatch):
    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError(f"boom {FAKE_KEY}")

    monkeypatch.setattr(g, "build_graph", lambda: Broken())
    at = ask(start(), "你好")
    assert not at.exception
    assert "抱歉，处理这个问题时出现了错误" in at.chat_message[1].markdown[0].value
    assert FAKE_KEY not in all_text(at)
    assert "Traceback" not in all_text(at)


def test_render_error_shows_friendly_message(monkeypatch):
    monkeypatch.setattr(g, "load_schools", lambda path=None: 1 / 0)
    at = start()
    assert not at.exception
    assert any("抱歉" in e.value for e in at.error)


def test_cached_schools_used_by_graph(monkeypatch):
    calls = []
    real = g.load_schools

    def counting(path=None):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(g, "load_schools", counting)
    at = ask(start(), "成均馆大学需要 GMAT 吗？")
    ask(at, "国民大学需要 GMAT 吗？")
    assert len(calls) == 1  # 数据由 st.cache_resource 缓存，节点不再重复读取文件


# ---------------------------------------------------------------------------
# export_graph.py
# ---------------------------------------------------------------------------

def test_export_png_failure_keeps_mmd(monkeypatch, tmp_path, capsys):
    from langchain_core.runnables.graph import Graph

    def fail(self, *a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(Graph, "draw_mermaid_png", fail)
    written = export_graph.export(tmp_path)
    assert written == [tmp_path / "graph.mmd"]
    assert "router" in (tmp_path / "graph.mmd").read_text(encoding="utf-8")
    assert not (tmp_path / "graph.png").exists()
    assert "PNG 导出失败（ConnectionError）" in capsys.readouterr().out


def test_export_png_success(monkeypatch, tmp_path):
    from langchain_core.runnables.graph import Graph

    monkeypatch.setattr(Graph, "draw_mermaid_png", lambda self, *a, **k: b"\x89PNG fake")
    written = export_graph.export(tmp_path)
    assert written == [tmp_path / "graph.mmd", tmp_path / "graph.png"]
    assert (tmp_path / "graph.png").read_bytes() == b"\x89PNG fake"
