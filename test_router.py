"""router 节点测试：模型分类与关键词降级（使用假模型，不调用真实 API）。"""

import json

import pytest
from langchain_core.exceptions import OutputParserException

import graph as g
from fake_llm import FakeLLM

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-321"


def use_llm(monkeypatch, response):
    fake = FakeLLM({g.RouterResult: response})
    monkeypatch.setattr(g, "get_llm", lambda: fake)
    return fake


def route(question):
    return g.router({"question": question})


def test_no_key_uses_keywords():
    result = route("帮我看看这篇文书")
    assert result["intent"] == "review"
    assert result["trace"] == ["router"]


@pytest.mark.parametrize("intent", g.INTENTS)
def test_llm_intent_used(monkeypatch, intent):
    use_llm(monkeypatch, g.RouterResult(intent=intent))
    result = route("任意问题")
    assert result["intent"] == intent
    assert result["trace"] == ["router", "router:llm"]


def test_llm_overrides_keywords(monkeypatch):
    # "规划" 在关键词版会命中 timeline；模型判断优先
    use_llm(monkeypatch, g.RouterResult(intent="profile"))
    assert route("我 GPA 3.6，想规划一下选校")["intent"] == "profile"


def test_llm_request(monkeypatch):
    fake = use_llm(monkeypatch, g.RouterResult(intent="qa"))
    route("成均馆大学要 GMAT 吗？")
    schema, method, (system, human) = fake.calls[0]
    assert schema is g.RouterResult and method == "json_schema"
    assert all(i in system.content for i in g.INTENTS)
    assert human.content == "成均馆大学要 GMAT 吗？"


@pytest.mark.parametrize("response, entry", [
    (g.RouterResult.model_construct(intent="weather"), "router:llm_invalid_intent"),
    (RuntimeError("down"), "router:llm_error"),
    (TimeoutError(), "router:llm_error"),
    (OutputParserException("bad json"), "router:llm_parse_error"),
    (None, "router:llm_parse_error"),
    ({"intent": "qa"}, "router:llm_parse_error"),
])
def test_llm_failure_falls_back_to_keywords(monkeypatch, response, entry):
    use_llm(monkeypatch, response)
    result = route("帮我看看这篇文书")
    assert result["intent"] == "review"  # 关键词版结果
    assert result["trace"] == ["router", entry]


def test_llm_error_does_not_leak_message(monkeypatch):
    use_llm(monkeypatch, RuntimeError(f"invalid key {FAKE_KEY}"))
    assert FAKE_KEY not in json.dumps(route("你好")["trace"])


def test_graph_chitchat_by_llm(monkeypatch):
    use_llm(monkeypatch, g.RouterResult(intent="chitchat"))
    result = g.run("早上好呀")
    assert result["trace"] == ["router", "router:llm"]
    assert "留学申请顾问" in result["messages"][-1].content


def test_graph_unclear_by_llm(monkeypatch):
    use_llm(monkeypatch, g.RouterResult(intent="unclear"))
    result = g.run("今天天气怎么样")
    assert result["trace"] == ["router", "router:llm", "clarify"]


def test_graph_router_error_still_answers(monkeypatch):
    use_llm(monkeypatch, RuntimeError("down"))
    result = g.run("帮我看看这篇文书")
    assert result["trace"] == ["router", "router:llm_error", "doc_reviewer", "doc_reviewer:too_short"]
