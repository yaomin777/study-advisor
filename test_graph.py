"""graph.py 的路由测试与 data/schools.json 格式检查。运行：.venv/bin/python -m pytest -q"""

import json
import re
from datetime import date

import pytest

import graph as g

FULL_PROFILE_Q = "我想选校，GPA 3.6，本科专业金融，雅思 7.0，GMAT 700"

# 路由测试使用独立的测试数据，不依赖真实学校数据
TODAY = date(2026, 10, 29)

TEST_SCHOOL = {
    "school": "TestSchool",
    "aliases": ["测试大学", "TS"],
    "program": "TEST MSc",
    "rounds": [
        {"name": "TEST Round 1", "deadline": "2026-10-28"},
        {"name": "TEST Round 2", "deadline": "2026-11-30"},
    ],
    "gmat_gre": "TEST",
    "language": "TEST",
    "background_req": "TEST",
    "essays": ["TEST"],
    "recommendations": "TEST",
    "source_url": "https://example.com/test",
    "as_of": "2000-01-01",
}


EXPIRED_SCHOOL = {
    **TEST_SCHOOL,
    "school": "OldSchool",
    "aliases": [],
    "rounds": [{"name": "OLD Round 1", "deadline": "2026-05-01"}],
}


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(g, "get_today", lambda: TODAY)


@pytest.fixture
def test_data(monkeypatch, tmp_path):
    path = tmp_path / "schools.json"
    path.write_text(json.dumps({"schools": [TEST_SCHOOL, EXPIRED_SCHOOL]}), encoding="utf-8")
    monkeypatch.setattr(g, "SCHOOLS_PATH", path)


# ---------------------------------------------------------------------------
# 图结构与路由
# ---------------------------------------------------------------------------

def test_structure():
    nodes = [n for n in g.graph.get_graph().nodes if n not in ("__start__", "__end__")]
    assert len(nodes) == 9
    conditional_sources = {e.source for e in g.graph.get_graph().edges if e.conditional}
    assert conditional_sources == {"router", "profile_extractor", "requirement_retriever", "verify"}


def test_chitchat():
    assert g.run("你好")["trace"] == ["router"]


def test_unclear():
    assert g.run("嗯……")["trace"] == ["router", "clarify"]


def test_review():
    assert g.run("帮我看看这篇文书")["trace"] == ["router", "doc_reviewer", "doc_reviewer:too_short"]


def test_profile_extraction():
    profile = g.run(FULL_PROFILE_Q)["profile"]
    assert profile == {"gpa": "3.6", "major": "金融", "language_score": "雅思 7.0", "gmat_gre": "GMAT 700"}


def test_profile_complete():
    result = g.run(FULL_PROFILE_Q)
    assert result["missing_fields"] == []
    assert result["trace"] == ["router", "profile_extractor", "school_matcher", "verify"]


def test_profile_missing():
    result = g.run("我想选校，GPA 3.5，GRE 320")
    assert result["profile"]["gmat_gre"] == "GRE 320"
    assert result["missing_fields"] == ["language_score", "major"]
    assert result["trace"] == ["router", "profile_extractor", "clarify"]


def test_gmat_gre_not_required():
    result = g.run("我想选校，GPA 3.6，本科专业金融，雅思 7.0")
    assert "gmat_gre" not in result["profile"]
    assert result["missing_fields"] == []
    assert result["trace"] == ["router", "profile_extractor", "school_matcher", "verify"]


def test_clarify_does_not_ask_gmat_gre():
    draft = g.run("我想选校，GPA 3.5")["draft"]
    assert "语言成绩" in draft and "本科专业" in draft
    assert "GMAT" not in draft and "GRE" not in draft


@pytest.mark.parametrize("question, field, value", [
    ("选校 GPA 3.6/4.3", "gpa", "3.6/4.3"),
    ("选校 GPA 85/100", "gpa", "85/100"),
    ("选校 TOPIK 4级", "language_score", "TOPIK 4"),
    ("选校 托福 95", "language_score", "托福 95"),
])
def test_profile_extraction_formats(question, field, value):
    assert g.run(question)["profile"][field] == value


def test_qa_with_source(test_data):
    result = g.run("TestSchool 的申请要求是什么？")
    assert result["trace"] == ["router", "requirement_retriever", "verify"]
    assert result["sources"] == ["https://example.com/test"]
    assert result["verdict"] == "pass"


def test_timeline_with_source(test_data):
    result = g.run("帮我做一个 testschool 的申请时间规划")
    assert result["trace"] == ["router", "requirement_retriever", "timeline_planner", "verify"]


def test_no_source_fallback(test_data):
    result = g.run("伦敦商学院的申请要求是什么？")
    assert result["trace"] == ["router", "requirement_retriever", "fallback"]


@pytest.mark.parametrize("question", ["测试大学申请要求", "TS 申请要求", "ts申请要求"])
def test_alias_match(test_data, question):
    result = g.run(question)
    assert result["trace"] == ["router", "requirement_retriever", "verify"]
    assert [d["school"] for d in result["docs"]] == ["TestSchool"]


@pytest.mark.parametrize("question", ["TSX 申请要求", "STS 申请要求"])
def test_ascii_alias_whole_word(test_data, question):
    assert g.run(question)["trace"][-1] == "fallback"


def test_round_status():
    assert g.round_status("2026-10-28", TODAY) == "已截止（以最新简章为准）"
    assert g.round_status("2026-10-29", TODAY) == "今日截止"
    assert g.round_status("2026-10-30", TODAY) == "未截止（距截止 1 天）"


def test_qa_answer_marks_expired(test_data):
    draft = g.run("TestSchool 申请要求")["draft"]
    assert "TEST Round 1 截止 2026-10-28，已截止（以最新简章为准）" in draft
    assert "TEST Round 2 截止 2026-11-30，未截止（距截止 32 天）" in draft


def test_timeline_marks_expired(test_data):
    draft = g.run("TestSchool 时间规划")["draft"]
    assert "今天是 2026-10-29" in draft
    assert "TEST Round 1 截止 2026-10-28，已截止" in draft
    assert "TEST Round 2 截止 2026-11-30，未截止" in draft
    assert "暂无可申请轮次" not in draft


def test_timeline_all_expired(test_data):
    draft = g.run("OldSchool 时间规划")["draft"]
    assert "OLD Round 1 截止 2026-05-01，已截止（以最新简章为准）" in draft
    assert "未截止" not in draft
    assert "暂无可申请轮次" in draft


def test_missing_data_file(tmp_path):
    assert g.load_schools(tmp_path / "none.json") == []


def _always_fail(state):
    return {"verdict": "fail", "trace": g._trace(state, "verify")}


def test_retry_then_fallback_qa(monkeypatch, test_data):
    monkeypatch.setattr(g, "verify", _always_fail)
    result = g.build_graph().invoke({"question": "TestSchool 申请要求"})
    assert result["trace"] == ["router"] + ["requirement_retriever", "verify"] * 3 + ["fallback"]
    assert result["retry_count"] == 2


def test_retry_then_fallback_school(monkeypatch):
    # 无密钥时选校回答已是模板理由：校验不通过直接 fallback，不再重写
    monkeypatch.setattr(g, "verify", _always_fail)
    result = g.build_graph().invoke({"question": FULL_PROFILE_Q})
    assert result["trace"] == ["router", "profile_extractor", "school_matcher", "verify", "fallback"]
    assert result["retry_count"] == 0


# ---------------------------------------------------------------------------
# data/schools.json 格式检查
# ---------------------------------------------------------------------------

SCHOOL_FIELDS = {
    "school", "aliases", "program", "rounds", "gmat_gre", "language", "background_req",
    "essays", "recommendations", "work_experience", "source_url", "as_of",
}
STRING_FIELDS = SCHOOL_FIELDS - {"aliases", "rounds", "essays"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date(value) -> bool:
    if not isinstance(value, str) or not DATE_RE.match(value):
        return False
    date.fromisoformat(value)  # 非法日期（如 02-30）会抛 ValueError
    return True


def test_schools_json_format():
    schools = g.load_schools()
    assert schools, "data/schools.json 中没有学校记录"
    for s in schools:
        name = s.get("school")
        assert set(s) == SCHOOL_FIELDS, f"{name}: 字段不符 {set(s) ^ SCHOOL_FIELDS}"
        for f in STRING_FIELDS:
            assert isinstance(s[f], str) and s[f].strip(), f"{name}: {f} 应为非空字符串"
        assert s["source_url"].startswith("https://"), f"{name}: source_url 应为 https 链接"
        assert _is_date(s["as_of"]), f"{name}: as_of 应为 YYYY-MM-DD"
        assert isinstance(s["aliases"], list) and s["aliases"], f"{name}: aliases 应为非空列表"
        assert all(isinstance(a, str) and a.strip() for a in s["aliases"]), f"{name}: aliases 含空值"
        assert len(s["aliases"]) == len(set(s["aliases"])), f"{name}: aliases 有重复"
        assert s["school"] not in s["aliases"], f"{name}: aliases 不应重复 school"
        assert isinstance(s["essays"], list) and all(isinstance(e, str) and e for e in s["essays"])
        assert isinstance(s["rounds"], list)
        for r in s["rounds"]:
            assert set(r) == {"name", "deadline"}, f"{name}: 轮次字段不符 {r}"
            assert r["name"].strip(), f"{name}: 轮次名为空"
            assert _is_date(r["deadline"]), f"{name}: deadline 应为 YYYY-MM-DD，实际 {r['deadline']}"
    names = [n.lower() for s in schools for n in [s["school"]] + s["aliases"]]
    assert len(names) == len(set(names)), "学校名称或别名在不同学校间重复"


def test_real_data_retrieval():
    result = g.run("成均馆大学的申请要求是什么？")
    assert result["trace"] == ["router", "requirement_retriever", "verify"]
    assert result["sources"] == ["https://gsb.skku.edu/en/Full-time_MBA/Admissions.do"]


def test_real_data_school_list():
    assert [s["school"] for s in g.load_schools()] == ["庆熙大学", "仁济大学", "成均馆大学", "国民大学", "崇实大学"]
    assert g.run("延世大学的申请要求是什么？")["trace"] == ["router", "requirement_retriever", "fallback"]


@pytest.mark.parametrize("question, school", [
    ("SKK GSB 申请要求", "成均馆大学"),
    ("인제대 申请要求", "仁济大学"),
    ("Kyung Hee University 申请要求", "庆熙大学"),
    ("kookmin 申请要求", "国民大学"),
    ("崇实申请要求", "崇实大学"),
])
def test_real_data_alias(question, school):
    assert [d["school"] for d in g.run(question)["docs"]] == [school]


FORBIDDEN_GMAT_PHRASES = ["不需要", "无需", "不要求", "不用"]


@pytest.mark.parametrize("school", [s["school"] for s in g.load_schools()])
def test_gmat_gre_answer_never_says_not_required(school):
    record = next(s for s in g.load_schools() if s["school"] == school)
    for question in (f"{school}需要GMAT吗？", f"{school} GRE 要求", f"{school}的申请要求"):
        draft = g.run(question)["draft"]
        if "未提及" in record["gmat_gre"]:
            assert f"GMAT/GRE {g.GMAT_GRE_UNMENTIONED}" in draft or f"GMAT/GRE：{g.GMAT_GRE_UNMENTIONED}" in draft
        assert not any(p in draft for p in FORBIDDEN_GMAT_PHRASES), f"{question} -> {draft}"


def test_gmat_gre_question_focused_answer():
    draft = g.run("仁济大学和成均馆大学需要 GMAT 吗？")["draft"]
    assert "仁济大学 经营大学院（2026学年度后期）：GMAT/GRE 官网未说明（核实日期 2026-10-29）（请以最新简章为准）" in draft
    assert "GMAT/GRE 选择性提交（GMAT或GRE成绩单），提交优秀成绩者在书面评审中加分（请以最新简章为准）" in draft
    assert "申请轮次" not in draft  # 只回答 GMAT/GRE，不输出全部要求


@pytest.mark.parametrize("school", [s["school"] for s in g.load_schools()])
def test_work_experience_answer_uses_source_text(school):
    record = next(s for s in g.load_schools() if s["school"] == school)
    result = g.run(f"{school}需要工作经验吗？")
    assert result["intent"] == "qa"
    assert f"工作经验 {record['work_experience']}" in result["draft"]
    assert "申请轮次" not in result["draft"]
    assert not any(p in result["draft"] for p in ["不接受", "不需要", "无需"])


def test_full_answer_includes_work_experience():
    draft = g.run("成均馆大学的申请要求是什么？")["draft"]
    assert "工作经验：报考资格中未要求工作经验" in draft


def test_gmat_and_work_experience_together():
    draft = g.run("国民大学需要 GMAT 和工作经验吗？")["draft"]
    assert "GMAT/GRE 官网未说明（核实日期 2026-10-29）（请以最新简章为准）；工作经验 报考资格中未要求工作经验" in draft


def test_real_data_no_false_alias_match():
    # "issue" 含 "ssu"、"injection" 含 "inje"，不应命中崇实/仁济
    assert g.run("申请要求 issue injection")["docs"] == []
