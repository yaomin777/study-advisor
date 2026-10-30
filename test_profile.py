"""profile_extractor 节点测试：模型抽取、防推测检查与正则降级（使用假模型，不调用真实 API）。"""

import json

import pytest
from langchain_core.exceptions import OutputParserException

import graph as g
from fake_llm import FakeLLM

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-654"
Q = "我本科学金融，GPA 3.6/4.0，雅思 7.0，想申请韩国的经营学硕士"


def P(gpa=None, language_score=None, gmat_gre=None, major=None):
    return g.ProfileResult(gpa=gpa, language_score=language_score, gmat_gre=gmat_gre, major=major)


def use_llm(monkeypatch, profile_response, intent="profile"):
    fake = FakeLLM({g.ProfileResult: profile_response, g.RouterResult: g.RouterResult(intent=intent)})
    monkeypatch.setattr(g, "get_llm", lambda: fake)
    return fake


def extract(question, profile=None):
    return g.profile_extractor({"question": question, "profile": profile or {}, "trace": ["router"]})


def test_llm_fields_used(monkeypatch):
    use_llm(monkeypatch, P(gpa="3.6/4.0", language_score="IELTS 7.0", major="金融"))
    result = extract(Q)
    assert result["profile"] == {
        "gpa": "3.6/4.0", "language_score": "IELTS 7.0", "major": "金融", "target_region": "韩国",
    }
    assert result["missing_fields"] == []
    assert result["trace"] == ["router", "profile_extractor", "profile_extractor:llm"]


def test_gmat_gre_optional(monkeypatch):
    use_llm(monkeypatch, P(gpa="3.6/4.0", language_score="IELTS 7.0", major="金融"))
    result = extract(Q)
    assert "gmat_gre" not in result["profile"]
    assert result["missing_fields"] == []


def test_gmat_gre_recorded_when_said(monkeypatch):
    use_llm(monkeypatch, P(gpa="3.6", language_score="TOEFL 95", gmat_gre="GMAT 700", major="会计"))
    result = extract("GPA 3.6，托福 95，GMAT 700，本科会计")
    assert result["profile"]["gmat_gre"] == "GMAT 700"


def test_missing_required_fields(monkeypatch):
    use_llm(monkeypatch, P(gpa="3.4"))
    result = extract("我GPA 3.4，想申请经营大学院")
    assert result["profile"] == {"gpa": "3.4"}
    assert result["missing_fields"] == ["language_score", "major"]


@pytest.mark.parametrize("question, value", [
    ("雅思7分，GPA 3.6，本科金融", "IELTS 7.0"),  # 7 与 7.0 视为同一数值
    ("TOPIK 4级，GPA 3.6，本科金融", "TOPIK 4"),
    ("托福 95，GPA 3.6，本科金融", "托福 95"),
])
def test_language_normalized_value_accepted(monkeypatch, question, value):
    use_llm(monkeypatch, P(gpa="3.6", language_score=value, major="金融"))
    result = extract(question)
    assert result["profile"]["language_score"] == value


@pytest.mark.parametrize("response, field", [
    (P(gpa="3.8", language_score="IELTS 7.0", major="金融"), "gpa"),                 # 数值与原话不符
    (P(gpa="3.6/4.0", language_score="IELTS 7.5", major="金融"), "language_score"),
    (P(gpa="3.6/4.0", language_score="TOEFL 7.0", major="金融"), "language_score"),  # 考试名与原话不符
    (P(gpa="3.6/4.0", language_score="IELTS 7.0", major="金融学"), "major"),          # 原话是"金融"，不得补全
    (P(gpa="3.6/4.0", language_score="IELTS 7.0", gmat_gre="GMAT 700", major="金融"), "gmat_gre"),  # 没说过
])
def test_guessed_value_dropped(monkeypatch, response, field):
    use_llm(monkeypatch, response)
    result = extract(Q)
    assert field not in result["profile"]
    assert f"profile_extractor:dropped({field})" in result["trace"]
    if field in g.REQUIRED_PROFILE_FIELDS:
        assert field in result["missing_fields"]


def test_previous_profile_kept_when_llm_returns_null(monkeypatch):
    use_llm(monkeypatch, P(language_score="IELTS 7.0"))
    result = extract("雅思 7.0", profile={"gpa": "3.6", "major": "金融"})
    assert result["profile"] == {"gpa": "3.6", "major": "金融", "language_score": "IELTS 7.0"}


def test_llm_request(monkeypatch):
    fake = use_llm(monkeypatch, P())
    extract(Q)
    schema, method, (system, human) = next(c for c in fake.calls if c[0] is g.ProfileResult)
    assert method == "json_schema"
    assert "null" in system.content and "不得推测" in system.content
    assert human.content == Q


@pytest.mark.parametrize("response, entry", [
    (RuntimeError("down"), "profile_extractor:llm_error"),
    (TimeoutError(), "profile_extractor:llm_error"),
    (OutputParserException("bad json"), "profile_extractor:llm_parse_error"),
    (None, "profile_extractor:llm_parse_error"),
    ({"gpa": "3.6"}, "profile_extractor:llm_parse_error"),
])
def test_llm_failure_falls_back_to_regex(monkeypatch, response, entry):
    use_llm(monkeypatch, response)
    result = extract("选校：GPA 3.6，本科专业金融，雅思 7.0")
    assert result["profile"] == {"gpa": "3.6", "major": "金融", "language_score": "雅思 7.0"}
    assert result["trace"] == ["router", "profile_extractor", entry]


def test_llm_error_does_not_leak_message(monkeypatch):
    use_llm(monkeypatch, RuntimeError(f"invalid key {FAKE_KEY}"))
    assert FAKE_KEY not in json.dumps(extract(Q)["trace"])


def test_graph_complete_profile(monkeypatch):
    fake = use_llm(monkeypatch, P(gpa="3.6/4.0", language_score="IELTS 7.0", major="金融"))
    fake.responses[g.MatchReasons] = g.MatchReasons(reasons=[])  # 理由全部退回模板
    fake.responses[g.VerifyResult] = g.VerifyResult(checks=[])
    result = g.run(Q)
    nodes = [t for t in result["trace"] if ":" not in t]
    assert nodes == ["router", "profile_extractor", "school_matcher", "verify"]
    assert "profile_extractor:llm" in result["trace"]
    assert result["verdict"] == "pass"


def test_graph_missing_fields_clarify_without_gmat(monkeypatch):
    use_llm(monkeypatch, P(gpa="3.4"))
    result = g.run("我GPA 3.4，想申请经营大学院")
    assert result["trace"][-1] == "clarify"
    assert "语言成绩" in result["draft"] and "本科专业" in result["draft"]
    assert "GMAT" not in result["draft"]


@pytest.mark.parametrize("value", ["TOPIK 4级", "TOPIK4", "TOPIK 4"])
def test_topik_formats_accepted(monkeypatch, value):
    use_llm(monkeypatch, P(gpa="3.2", language_score=value, major="计算机"))
    result = extract("本科计算机，GPA 3.2，TOPIK 4级")
    assert result["profile"]["language_score"] == value
