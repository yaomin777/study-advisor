"""verify 节点测试：代码核对（日期、否定说法）和模型核对（使用假模型，不调用真实 API）。"""

import json
from datetime import date

import pytest
from langchain_core.exceptions import OutputParserException

import graph as g
from fake_llm import FakeLLM

TODAY = date(2026, 10, 30)
FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-789"
OK = g.VerifyResult(checks=[])


def V(*checks):
    """V((1, ["依据原文", ...]), (2, [])) -> VerifyResult；有无依据由代码检查 evidence 决定"""
    return g.VerifyResult(checks=[g.ClaimCheck(index=i, evidence=ev, reason="说明") for i, ev in checks])


EV_COMPARE = ["TOPIK 5级以上", "TOPIK 6"]  # 学校记录一侧 + 用户画像一侧

DOC = {
    "school": "TestSchool",
    "aliases": ["TS"],
    "program": "TEST MBA",
    "rounds": [{"name": "报名 2026-10-01 起", "deadline": "2026-11-06"}],
    "gmat_gre": "官网未说明（核实日期 2026-10-29）",
    "language": "TOPIK 5级以上",
    "background_req": "未限制专业",
    "essays": ["自我介绍书"],
    "recommendations": "官网未说明（核实日期 2026-10-29）",
    "work_experience": "报考资格中未要求工作经验",
    "source_url": "https://example.com/test",
    "as_of": "2026-10-29",
}


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(g, "get_today", lambda: TODAY)


def use_llm(monkeypatch, responses):
    fake = FakeLLM(responses)
    monkeypatch.setattr(g, "get_llm", lambda: fake)
    return fake


def state(draft, docs=(DOC,), last_agent="requirement_retriever"):
    docs = list(docs)
    return {
        "draft": draft,
        "docs": docs,
        "sources": [d["source_url"] for d in docs],
        "last_agent": last_agent,
        "trace": ["router", last_agent],
    }


def reasons(result):
    return [t for t in result["trace"] if t.startswith("verify:")]


# ---------------------------------------------------------------------------
# a 层：日期核对（无密钥时也执行）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("draft", [
    "截止 2026-11-06", "截止 2026年11月6日", "截止 2026.11.06", "截止 2026/11/6",
    "11月6日截止", "核实日期 2026-10-29", "报名 2026-10-01 起",
    "今天是 2026-10-30",  # 当天日期允许出现
    "没有日期的回答",
])
def test_dates_in_sources_pass(draft):
    result = g.verify(state(draft))
    assert result["verdict"] == "pass"
    assert reasons(result) == []


@pytest.mark.parametrize("draft", [
    "截止 2026-11-07", "截止 2026年11月7日", "截止 2026.11.07", "截止 2026/11/7",
    "11月7日截止", "截止 2026-02-30", "截止 2027-11-06",
])
def test_date_not_in_sources_fail(draft):
    result = g.verify(state(draft))
    assert result["verdict"] == "fail"
    assert "verify:dates_not_in_sources" in reasons(result)


def test_date_without_any_docs_fails():
    result = g.verify(state("截止 2026-11-06", docs=(), last_agent="school_matcher"))
    assert result["verdict"] == "fail"
    assert "verify:dates_not_in_sources" in reasons(result)


def test_unsupported_dates_lists_each_bad_date():
    bad = g.unsupported_dates("2026-11-06、2026-12-01、3月1日", [DOC], TODAY)
    assert bad == ["2026-12-01", "3月1日"]


# ---------------------------------------------------------------------------
# d：“官网未说明”“未提及”不能变成“不要求”
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("draft", [
    "TestSchool GMAT/GRE：不需要", "推荐信：不要求", "无需提交推荐信", "不接受有工作经验者",
])
def test_negation_without_source_fails(draft):
    result = g.verify(state(draft))
    assert result["verdict"] == "fail"
    assert "verify:negation_not_in_sources" in reasons(result)


def test_negation_quoted_from_source_passes():
    doc = {**DOC, "recommendations": "硕士不需要提交推荐信"}
    result = g.verify(state("推荐信：硕士不需要提交推荐信", docs=(doc,)))
    assert result["verdict"] == "pass"


def test_unmentioned_wording_passes():
    result = g.verify(state(f"GMAT/GRE {g.GMAT_GRE_UNMENTIONED}；工作经验 报考资格中未要求工作经验"))
    assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 待核对陈述的提取：原文子串由代码判定，评论性语句不列入
# ---------------------------------------------------------------------------

COMPARE = "语言要求 TOPIK 5级以上，你的 TOPIK 6级达到"


@pytest.mark.parametrize("draft", [
    "语言：TOPIK 5级以上",
    "语言要求：TOPIK 5级以上。",                 # 标点差异
    "- TestSchool TEST MBA：GMAT/GRE 官网未说明（核实日期 2026-10-29）（请以最新简章为准）（出处：https://example.com/test，截至 2026-10-29）",
    "工作经验 报考资格中未要求工作经验",
    "TestSchool：推荐信：官网未说明（核实日期2026-10-29）",  # 空白差异
])
def test_verbatim_statements_not_sent_to_llm(draft):
    assert g.verify_statements(draft, [DOC], "requirement_retriever") == []


@pytest.mark.parametrize("draft", ["与记录一致", "总体来看材料比较齐全", "申请轮次（今天 2026-10-30）：报名 截止 2026-11-06", "根据检索到的资料："])
def test_commentary_and_code_lines_not_statements(draft):
    assert g.verify_statements(draft, [DOC], "requirement_retriever") == []


def test_non_verbatim_key_statement_sent():
    assert g.verify_statements(COMPARE, [DOC], "requirement_retriever") == [COMPARE]


def test_full_answer_statements_prefixed_with_school():
    draft = "根据检索到的资料：\nTestSchool TEST MBA\n  语言：TOPIK 4级以上\n  文书：自我介绍书"
    assert g.verify_statements(draft, [DOC], "requirement_retriever") == ["TestSchool：语言：TOPIK 4级以上"]


def test_school_matcher_sends_only_supplements():
    draft = "\n".join([
        "你的画像：GPA 3.2（学校记录未提供 GPA 要求，未作判断）；语言成绩 TOPIK 6；本科专业 计算机；GMAT/GRE 未提供",
        g.MATCH_HEADER,
        "",
        "TestSchool TEST MBA",
        "  语言：符合（要求 TOPIK 5级以上，你的 TOPIK 6级）｜原文：TOPIK 5级以上",
        "  本科专业：未限制专业｜原文：未限制专业",
        f"  补充说明：{COMPARE}；未限制专业",
        "  出处：https://example.com/test（数据整理日期 2026-10-29）",
    ])
    assert g.verify_statements(draft, [DOC], "school_matcher") == [f"TestSchool：{COMPARE}"]


@pytest.mark.parametrize("evidence, parts", [
    ("language: TOPIK 5级以上", ["TOPIK 5级以上"]),
    ('"background_req": "未限制专业"', ["未限制专业"]),
    ("major：计算机", ["计算机"]),
    ('language": "TOPIK 5级以上', ["TOPIK 5级以上"]),
    ('{"language": "TOPIK 5级以上", "major": "计算机"}', ["TOPIK 5级以上", "计算机"]),
    ("TOPIK 5级以上", ["TOPIK 5级以上"]),
    ('name": "报名 2026-10-01 起","deadline": "2026-11-06', ["报名 2026-10-01 起", "2026-11-06"]),  # 轮次里的嵌套字段
    ('{"name": "报名 2026-10-01 起", "deadline": "2026-11-06"}', ["报名 2026-10-01 起", "2026-11-06"]),
])
def test_evidence_parts_strip_field_names(evidence, parts):
    assert g.evidence_parts(evidence, g._all_keys([DOC]) | set(g.PROFILE_FIELDS)) == parts


def test_all_keys_include_nested_round_fields():
    assert {"name", "deadline", "language", "source_url"} <= g._all_keys([DOC])


def test_nested_field_evidence_accepted_by_verify(monkeypatch):
    draft = "TestSchool：报名 2026-10-01 起，你的 TOPIK 6级达到语言要求"
    use_llm(monkeypatch, {g.VerifyResult: V((1, ['name": "报名 2026-10-01 起","deadline": "2026-11-06', "TOPIK 6"]))})
    assert g.verify({**state(draft), "profile": PROFILE})["verdict"] == "pass"


# ---------------------------------------------------------------------------
# b 层：模型逐条判断（假模型）
# ---------------------------------------------------------------------------

PROFILE = {"gpa": "3.2", "language_score": "TOPIK 6", "major": "计算机", "target_region": "韩国"}


def test_no_key_skips_llm():
    result = g.verify(state(COMPARE))
    assert result["verdict"] == "pass"
    assert result["trace"][-1] == "verify"


def test_all_verbatim_no_llm_call(monkeypatch):
    fake = use_llm(monkeypatch, {g.VerifyResult: OK})
    result = g.verify(state("语言：TOPIK 5级以上"))
    assert result["verdict"] == "pass"
    assert reasons(result) == ["verify:no_claims"]
    assert fake.calls == []


def test_llm_pass(monkeypatch):
    fake = use_llm(monkeypatch, {g.VerifyResult: V((1, EV_COMPARE))})
    result = g.verify({**state(COMPARE), "profile": PROFILE})
    assert result["verdict"] == "pass"
    assert reasons(result) == ["verify:llm_pass"]
    schema, method, (system, human) = fake.calls[0]
    assert schema is g.VerifyResult and method == "json_schema"
    text = human.content
    assert text.index("【学校记录】") < text.index("【用户画像】") < text.index("【待核对陈述】")
    assert text.endswith(f"【待核对陈述】\n1. {COMPARE}")
    user_part = text[text.index("【用户画像】"):text.index("【待核对陈述】")]
    assert '"language_score": "TOPIK 6"' in user_part and "target_region" not in user_part
    assert DOC["source_url"] in text and '"aliases"' not in text
    for rule in ("比较方向", "录取概率", "官网未说明", "每条一次", "逐字摘录", "evidence"):
        assert rule in system.content


def test_verify_uses_larger_max_tokens(monkeypatch):
    fake = FakeLLM({g.VerifyResult: V((1, EV_COMPARE))})
    requested = []

    class Model:
        def with_max_tokens(self, n):
            requested.append(n)
            return fake

    monkeypatch.setattr(g, "get_llm", lambda: Model())
    result = g.verify({**state(COMPARE), "profile": PROFILE})
    assert reasons(result) == ["verify:llm_pass"]
    assert requested == [2048] and len(fake.calls) == 1


def test_empty_profile_still_labeled(monkeypatch):
    fake = use_llm(monkeypatch, {g.VerifyResult: OK})
    g.verify(state(COMPARE))
    assert "【用户画像】\n{}" in fake.calls[0][2][1].content


def test_unsupported_claims_counted_and_traced(monkeypatch):
    draft = f"{COMPARE}\n文书：两篇英文 essay"
    use_llm(monkeypatch, {g.VerifyResult: V((1, []), (2, ["两篇英文 essay"]))})
    result = g.verify({**state(draft), "profile": PROFILE})
    assert result["verdict"] == "fail"
    assert reasons(result) == [
        "verify:llm_unsupported_claims(2)",
        f"verify:unsupported:{COMPARE}｜未给出依据",
        "verify:unsupported:文书：两篇英文 essay｜依据不在资料中：两篇英文 essay",
    ]
    assert "messages" not in result


def test_evidence_found_passes(monkeypatch):
    draft = f"{COMPARE}\n文书：两篇英文 essay"
    use_llm(monkeypatch, {g.VerifyResult: V((1, EV_COMPARE), (2, ["自我介绍书"]))})
    assert g.verify({**state(draft), "profile": PROFILE})["verdict"] == "pass"


@pytest.mark.parametrize("evidence", [
    ["TOPIK 5 级以上。", "TOPIK6"],        # 空白、标点差异
    ["topik 5级以上", "topik 6"],          # 大小写差异
    ["language: TOPIK 5级以上", 'language_score": "TOPIK 6'],  # 带字段名和 JSON 标点
])
def test_evidence_normalized_match(monkeypatch, evidence):
    use_llm(monkeypatch, {g.VerifyResult: V((1, evidence))})
    assert g.verify({**state(COMPARE), "profile": PROFILE})["verdict"] == "pass"


@pytest.mark.parametrize("evidence", [
    ["TOPIK 4级以上", "TOPIK 6"],          # 学校一侧被改写
    ["TOPIK 5级以上", "TOPIK 7"],          # 用户一侧不在画像中
    ["TOPIK 5级以上", "韩国"],             # target_region 未作为资料提供
])
def test_fabricated_evidence_fails(monkeypatch, evidence):
    use_llm(monkeypatch, {g.VerifyResult: V((1, evidence))})
    result = g.verify({**state(COMPARE), "profile": PROFILE})
    assert result["verdict"] == "fail"
    assert "依据不在资料中" in reasons(result)[1]


def test_model_cannot_mark_supported_without_evidence(monkeypatch):
    # 旧做法采信模型的 true/false；现在没有 evidence 一律判为没有依据
    use_llm(monkeypatch, {g.VerifyResult: V((1, ["   "]))})
    assert g.verify({**state(COMPARE), "profile": PROFILE})["verdict"] == "fail"


def test_out_of_range_index_ignored(monkeypatch):
    use_llm(monkeypatch, {g.VerifyResult: V((1, EV_COMPARE), (9, []))})
    assert g.verify({**state(COMPARE), "profile": PROFILE})["verdict"] == "pass"


def test_long_claim_truncated(monkeypatch):
    long_draft = "语言要求" + "很长" * 60
    use_llm(monkeypatch, {g.VerifyResult: V((1, []))})
    result = g.verify(state(long_draft))
    assert reasons(result)[1] == f"verify:unsupported:{long_draft[:g.MAX_CLAIM_CHARS]}｜未给出依据"


@pytest.mark.parametrize("response, reason", [
    (RuntimeError("down"), "verify:llm_error"),
    (TimeoutError(), "verify:llm_error"),
    (OutputParserException("bad json"), "verify:llm_parse_error"),
    (None, "verify:llm_parse_error"),
    ({"checks": []}, "verify:llm_parse_error"),
])
def test_llm_failure_counts_as_fail(monkeypatch, response, reason):
    use_llm(monkeypatch, {g.VerifyResult: response})
    result = g.verify(state(COMPARE))
    assert result["verdict"] == "fail"
    assert reasons(result) == [reason]


def test_llm_error_does_not_leak_message(monkeypatch):
    use_llm(monkeypatch, {g.VerifyResult: RuntimeError(f"invalid key {FAKE_KEY}")})
    result = g.verify(state(COMPARE))
    assert FAKE_KEY not in json.dumps(result["trace"])


def test_llm_not_called_when_code_check_fails(monkeypatch):
    fake = use_llm(monkeypatch, {g.VerifyResult: OK})
    g.verify(state(f"{COMPARE}，截止 2026-12-31"))
    assert fake.calls == []


def test_llm_not_called_without_docs(monkeypatch):
    fake = use_llm(monkeypatch, {})
    result = g.verify(state(COMPARE, docs=(), last_agent="school_matcher"))
    assert result["verdict"] == "pass"
    assert fake.calls == []


@pytest.mark.parametrize("draft", ["你很可能录取", "录取把握大", "录取机会较高", "这所学校稳录"])
def test_admission_claim_fails_without_llm(monkeypatch, draft):
    fake = use_llm(monkeypatch, {g.VerifyResult: OK})
    result = g.verify(state(draft))
    assert result["verdict"] == "fail"
    assert "verify:admission_claim" in reasons(result)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# 整图：假模型下的流转（router 也由假模型给出 intent）
# 庆熙大学的 GMAT/GRE 记录为"官网未提及"，回答改写为"资料中未提及……"，不是原文子串，会交给模型
# ---------------------------------------------------------------------------

QA = {g.RouterResult: g.RouterResult(intent="qa")}
Q_KHU = "庆熙大学需要 GMAT 吗？"


def test_graph_llm_pass(monkeypatch):
    use_llm(monkeypatch, {**QA, g.VerifyResult: V((1, ["官网未提及"]))})
    result = g.run(Q_KHU)
    assert result["trace"] == ["router", "router:llm", "requirement_retriever", "verify", "verify:llm_pass"]


def test_graph_verbatim_answer_needs_no_verify_llm(monkeypatch):
    fake = use_llm(monkeypatch, QA)
    result = g.run("成均馆大学需要 GMAT 吗？")
    assert result["trace"] == ["router", "router:llm", "requirement_retriever", "verify", "verify:no_claims"]
    assert len(fake.calls) == 1


@pytest.mark.parametrize("error", [RuntimeError("down"), TimeoutError()])
def test_graph_llm_error_goes_straight_to_fallback(monkeypatch, error):
    fake = use_llm(monkeypatch, {**QA, g.VerifyResult: error})
    result = g.run(Q_KHU)
    assert result["trace"] == [
        "router", "router:llm", "requirement_retriever", "verify", "verify:llm_error", "fallback",
    ]
    assert result["retry_count"] == 0
    assert len(fake.calls_for(g.VerifyResult)) == 1


def test_graph_unsupported_claims_still_retry(monkeypatch):
    fake = use_llm(monkeypatch, {**QA, g.VerifyResult: V((1, []))})
    result = g.run(Q_KHU)
    assert result["trace"][-1] == "fallback"
    assert result["retry_count"] == 2
    assert len(fake.calls_for(g.VerifyResult)) == 3


def test_graph_out_of_library_never_calls_verify_llm(monkeypatch):
    fake = use_llm(monkeypatch, QA)
    result = g.run("首尔大学的申请要求是什么？")
    assert result["trace"] == ["router", "router:llm", "requirement_retriever", "fallback"]
    assert fake.calls_for(g.VerifyResult) == []
