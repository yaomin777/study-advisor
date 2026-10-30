"""app.py（Streamlit 界面）与 export_graph.py 测试。用 AppTest 无头运行，不调用真实模型。"""

import re
from pathlib import Path

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
    assert "1. `router`\n2. `requirement_retriever`\n3. `verify`" in sidebar
    assert "[成均馆大学](https://gsb.skku.edu/en/Full-time_MBA/Admissions.do)（整理日期 2026-10-29）" in sidebar
    assert "数据整理日期：** 2026-10-29" in sidebar


def test_node_path_filters_internal_entries():
    at = start()
    at.session_state["last_result"] = {"trace": ["router", "router:llm", "requirement_retriever", "verify", "verify:llm_pass"]}
    at.run()
    path = [m.value for m in at.sidebar.markdown if m.value.startswith("1. ")]
    assert path == ["1. `router`\n2. `requirement_retriever`\n3. `verify`"]
    expander = at.sidebar.expander[0]
    assert expander.label == "完整 trace"
    assert [c.value for c in expander.code] == ["router → router:llm → requirement_retriever → verify → verify:llm_pass"]


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


# ---------------------------------------------------------------------------
# 显示：出处链接、换行、节点路径
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r"\[((?:\\.|[^\]\\])*)\]\(([^)\s]*)\)")
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


def rendered_markdown(at: AppTest) -> list[str]:
    return [str(m.value) for m in at.markdown] + [
        str(m.value) for msg in at.chat_message for m in msg.markdown
    ] + [str(m.value) for m in at.sidebar.markdown]


def check_links(texts: list[str]) -> int:
    """每个链接的网址部分只含 ASCII（无中文、无全角标点）；链接之外不再有裸网址。返回链接数。"""
    count = 0
    for text in texts:
        for _, url in LINK_RE.findall(text):
            count += 1
            assert url.startswith("http") and not NON_ASCII_RE.search(url), url
        assert "http" not in LINK_RE.sub("", text), text
    return count


@pytest.mark.parametrize("question", [
    "成均馆大学需要 GMAT 吗？",                                # 问答：（出处：网址，截至 …）
    "我本科学金融，GPA 3.6/4.0，雅思 7.0，帮我选校",            # 选校：出处：网址（数据整理日期 …）
    "我想申请国民大学，请帮我规划申请时间",                     # 时间规划
    "庆熙大学的申请要求是什么？",                               # 完整要求：出处：网址（截至 …）
])
def test_rendered_links_are_ascii_only(question):
    at = ask(start(), question)
    assert not at.exception
    assert check_links(rendered_markdown(at)) >= 2  # 回答区和侧边栏都有链接


def test_date_note_outside_link():
    app = load_app_helpers()
    text = app.link_urls("出处：https://gsb.skku.edu/en/Full-time_MBA/Admissions.do（截至 2026-10-29）")
    assert text == "出处：[https://gsb.skku.edu/en/Full-time\\_MBA/Admissions.do](https://gsb.skku.edu/en/Full-time_MBA/Admissions.do)（截至 2026-10-29）"
    assert app.link_urls("（出处：https://a.example/x?y=1&z=2，截至 2026-10-29）") == (
        "（出处：[https://a.example/x?y=1&z=2](https://a.example/x?y=1&z=2)，截至 2026-10-29）"
    )
    assert app.link_urls("see https://a.example/x.") == "see [https://a.example/x](https://a.example/x)."


def load_app_helpers():
    """只加载 app.py 中的显示函数（不执行页面）。"""
    import types
    module = types.ModuleType("app_helpers")
    exec(Path("app.py").read_text(encoding="utf-8").split("@st.cache_resource")[0], module.__dict__)
    return module


def test_format_answer_keeps_line_breaks():
    app = load_app_helpers()
    answer = "根据检索到的资料：\n庆熙大学 经营大学院\n  语言：TOPIK 5级以上\n  出处：https://x.example/a（截至 2026-10-29）\n\n结尾一行\n第二行"
    out = app.format_answer(answer)
    assert out == (
        "根据检索到的资料：  \n庆熙大学 经营大学院\n- 语言：TOPIK 5级以上\n"
        "- 出处：[https://x.example/a](https://x.example/a)（截至 2026-10-29）"
        "\n\n结尾一行  \n第二行"
    )


def test_list_followed_by_plain_line_is_separated():
    app = load_app_helpers()
    assert app.format_answer("标题\n  字段：一\n说明行") == "标题\n- 字段：一\n\n说明行"


def test_answer_rendered_line_by_line():
    at = ask(start(), "我本科学金融，GPA 3.6/4.0，雅思 7.0，帮我选校")
    shown = at.chat_message[1].markdown[0].value
    assert "\n- 语言：" in shown and "\n- 本科专业：" in shown and "\n- 截止状态：" in shown
    raw = at.session_state["history"][1]["content"]
    assert "\n  语言：" in raw  # 保存的回答内容本身不变


def test_node_path_shows_every_node_including_last():
    at = start()
    trace = ["router", "requirement_retriever", "verify", "verify:dates_not_in_sources", "requirement_retriever",
             "verify", "verify:dates_not_in_sources", "requirement_retriever", "verify", "fallback"]
    at.session_state["last_result"] = {"trace": trace}
    at.run()
    path = next(m.value for m in at.sidebar.markdown if m.value.startswith("1. "))
    lines = path.splitlines()
    assert len(lines) == 8 and lines[-1] == "8. `fallback`"
    assert at.sidebar.expander[0].code[0].proto.wrap_lines
