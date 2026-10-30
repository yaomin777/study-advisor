"""doc_reviewer 节点测试：长度检查、模型点评与规则版降级（使用假模型，不调用真实 API）。"""

import json

import pytest
from langchain_core.exceptions import OutputParserException

import graph as g
from fake_llm import FakeLLM

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-111"
ESSAY = (
    "我本科就读于工商管理专业，大三在一家物流公司实习时，负责整理仓储周报，"
    "发现库存周转数据长期没有被用于决策。我用三周时间整理了过去一年的出入库记录，"
    "提出调整补货周期的建议，被部门采纳后试行了两个月。这段经历让我意识到，"
    "经营决策需要数据分析与管理视角的结合。我希望在经营学硕士阶段系统学习运营管理和数据分析，"
    "毕业后从事供应链管理相关工作，并希望将来回到中韩贸易相关的企业，帮助企业改进库存与采购流程。"
)
REQUEST = f"帮我看看这篇文书：{ESSAY}"
REVIEW = g.ReviewResult(structure="结构清楚。", relevance="与项目有联系。", specificity="有实习细节。")


def use_llm(monkeypatch, review_response, intent="review"):
    fake = FakeLLM({g.ReviewResult: review_response, g.RouterResult: g.RouterResult(intent=intent)})
    monkeypatch.setattr(g, "get_llm", lambda: fake)
    return fake


def review(question):
    return g.doc_reviewer({"question": question, "trace": ["router"]})


def test_essay_text_strips_request_prefix():
    assert g.essay_text(REQUEST) == ESSAY
    assert g.essay_text(ESSAY) == ESSAY


def test_essay_is_long_enough():
    assert len(ESSAY) >= g.MIN_ESSAY_CHARS


@pytest.mark.parametrize("question", ["帮我看看这篇文书", "帮我点评一下：我想读MBA，因为我喜欢管理。"])
def test_too_short_asks_for_more_without_calling_llm(monkeypatch, question):
    fake = use_llm(monkeypatch, REVIEW)
    result = review(question)
    assert "请粘贴完整的段落或全文" in result["draft"]
    assert result["trace"] == ["router", "doc_reviewer", "doc_reviewer:too_short"]
    assert fake.calls == []


def test_rule_version_without_key():
    result = review(REQUEST)
    assert result["trace"] == ["router", "doc_reviewer"]
    for section in ("【结构】", "【针对性】", "【具体性】"):
        assert section in result["draft"]
    assert "规则版" in result["draft"]


def test_llm_feedback(monkeypatch):
    use_llm(monkeypatch, REVIEW)
    result = review(REQUEST)
    assert result["draft"] == (
        "文书反馈（仅供参考，未改写原文）：\n【结构】结构清楚。\n【针对性】与项目有联系。\n【具体性】有实习细节。"
    )
    assert result["trace"] == ["router", "doc_reviewer", "doc_reviewer:llm"]
    assert result["messages"][-1].content == result["draft"]


def test_llm_request(monkeypatch):
    fake = use_llm(monkeypatch, REVIEW)
    review(REQUEST)
    schema, method, (system, human) = fake.calls[0]
    assert schema is g.ReviewResult and method == "json_schema"
    assert "不改写全文" in system.content
    assert all(w in system.content for w in ("结构", "针对性", "具体性"))
    assert human.content == ESSAY  # 只发送正文


@pytest.mark.parametrize("response, entry", [
    (RuntimeError("down"), "doc_reviewer:llm_error"),
    (TimeoutError(), "doc_reviewer:llm_error"),
    (OutputParserException("bad json"), "doc_reviewer:llm_parse_error"),
    (None, "doc_reviewer:llm_parse_error"),
    (g.ReviewResult(structure="  ", relevance="x", specificity="y"), "doc_reviewer:llm_parse_error"),
])
def test_llm_failure_falls_back_to_rules(monkeypatch, response, entry):
    use_llm(monkeypatch, response)
    result = review(REQUEST)
    assert result["trace"] == ["router", "doc_reviewer", entry]
    assert "规则版" in result["draft"]


def test_llm_error_does_not_leak_message(monkeypatch):
    use_llm(monkeypatch, RuntimeError(f"invalid key {FAKE_KEY}"))
    result = review(REQUEST)
    assert FAKE_KEY not in result["draft"] + json.dumps(result["trace"])


def test_graph_review_by_llm(monkeypatch):
    use_llm(monkeypatch, REVIEW)
    result = g.run(REQUEST)
    assert result["trace"] == ["router", "router:llm", "doc_reviewer", "doc_reviewer:llm"]
