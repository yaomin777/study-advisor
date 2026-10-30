"""school_matcher 节点测试：逐校要求符合度（代码判断）+ 模型补充说明（verify 核对，未通过就不显示）。
使用假模型，不调用真实 API。"""

import json
from datetime import date

import pytest
from langchain_core.exceptions import OutputParserException

import graph as g
from fake_llm import FakeLLM

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-222"
TODAY = date(2026, 10, 30)
FIT_WORDS = (g.FIT_YES, g.FIT_NO, g.FIT_UNKNOWN, g.FIT_UNSTATED)

TOPIK_U = {
    "school": "TopikU", "aliases": ["TU"], "program": "MBA",
    "rounds": [{"name": "R1", "deadline": "2026-05-01"}, {"name": "R2", "deadline": "2026-11-13"}],
    "language": "TOPIK 5级以上（仅限外国人）", "background_req": "4年制大学毕业（预定）者，未限制专业",
    "gmat_gre": "官网未提及", "work_experience": "报考资格中未要求工作经验",
    "source_url": "https://example.com/topik", "as_of": "2026-10-29",
}
MIX_U = {
    "school": "MixU", "program": "硕士", "rounds": [],
    "language": "外国人特别录取：TOPIK 3级以上，或英语能力成绩可替代（具体分数以简章为准）；入学后毕业需 TOPIK 4级以上",
    "background_req": "与所修专业无关均可报考",
    "source_url": "https://example.com/mix", "as_of": "2026-10-29",
}
ENG_U = {
    "school": "EngU", "program": "Full-Time MBA", "rounds": [],
    "language": "TOEFL、TOEIC、IELTS 中选其一", "background_req": "学士学位取得者",
    "gmat_gre": "选择性提交", "source_url": "https://example.com/eng", "as_of": "2026-10-29",
}
NODATA_U = {
    "school": "NoDataU", "program": "MBA", "rounds": [],
    "language": "官网未说明（核实日期 2026-10-29）",
    "source_url": "https://example.com/nodata", "as_of": "2026-10-29",
}
SCHOOLS = [TOPIK_U, MIX_U, ENG_U, NODATA_U]
CONFIG = {"configurable": {"schools": SCHOOLS}}
PROFILE = {"gpa": "3.2", "language_score": "TOPIK 6级", "major": "计算机"}


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(g, "get_today", lambda: TODAY)


def R(**reasons):
    return g.MatchReasons(reasons=[g.SchoolReason(school=k, reason=v) for k, v in reasons.items()])


def use_llm(monkeypatch, response):
    fake = FakeLLM({g.MatchReasons: response})
    monkeypatch.setattr(g, "get_llm", lambda: fake)
    return fake


def match(profile=PROFILE, config=CONFIG):
    return g.school_matcher({"profile": profile, "trace": ["router", "profile_extractor"]}, config)


def block(draft: str, school: str) -> str:
    """回答中某所学校的段落。"""
    return next(p for p in draft.split("\n\n") if p.startswith(f"{school} "))


# ---------------------------------------------------------------------------
# 语言成绩解析与 TOPIK 最低级别
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score, parsed", [
    ("TOPIK 4级", ("TOPIK", 4.0)), ("TOPIK4", ("TOPIK", 4.0)), ("4级", ("TOPIK", 4.0)), ("4 级", ("TOPIK", 4.0)),
    ("TOPIK3级", ("TOPIK", 3.0)), ("topik 6", ("TOPIK", 6.0)), ("雅思7分", ("IELTS", 7.0)),
    ("IELTS 7.0", ("IELTS", 7.0)), ("托福 95", ("TOEFL", 95.0)), ("7分", None), ("", None),
])
def test_parse_language_score(score, parsed):
    assert g.parse_language_score(score) == parsed


@pytest.mark.parametrize("record, minimum", [(TOPIK_U, 5), (MIX_U, 3), (ENG_U, None), (NODATA_U, None)])
def test_topik_minimum_ignores_graduation_requirement(record, minimum):
    assert g.topik_minimum(record["language"]) == minimum


# ---------------------------------------------------------------------------
# 语言符合度（代码判断）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language_score, record, expected", [
    ("TOPIK 6级", TOPIK_U, "符合（要求 TOPIK 5级以上，你的 TOPIK 6级）"),
    ("TOPIK 5", TOPIK_U, "符合（要求 TOPIK 5级以上，你的 TOPIK 5级）"),
    ("TOPIK 3级", TOPIK_U, "不符合（要求 TOPIK 5级以上，你的 TOPIK 3级）"),
    ("TOPIK 3级", MIX_U, "符合（要求 TOPIK 3级以上，你的 TOPIK 3级）"),   # 毕业要求 4级 不计
    ("TOPIK 2级", MIX_U, "不符合（要求 TOPIK 3级以上，你的 TOPIK 2级）"),
    ("TOEFL 95", ENG_U, "无法判断，以简章为准（记录未给出 TOEFL 的具体分数要求）"),
    ("IELTS 7.0", MIX_U, "无法判断，以简章为准（记录未给出 IELTS 的具体分数要求）"),
    ("TOEFL 95", TOPIK_U, "无法判断（要求 TOPIK 5级以上，未提供 TOPIK 成绩）"),
    ("TOPIK 6级", ENG_U, "无法判断，以简章为准（记录未列出 TOPIK 级别要求）"),
    ("TOPIK 6级", NODATA_U, "官网未说明"),
    ("", TOPIK_U, "无法判断（未提供可识别的语言成绩）"),
])
def test_language_fit(language_score, record, expected):
    result = g.language_fit(record, {"language_score": language_score})
    assert result == expected
    assert result.startswith(FIT_WORDS)


def test_language_missing_in_record_is_unstated():
    assert g.language_fit({"school": "X"}, PROFILE) == "官网未说明"


@pytest.mark.parametrize("record, expected", [
    (TOPIK_U, "未限制专业"), (MIX_U, "未限制专业"), (ENG_U, "官网未说明"), (NODATA_U, "官网未说明"),
])
def test_major_fit(record, expected):
    assert g.major_fit(record) == expected


def test_school_fit_copies_record_text_and_deadline_status():
    fit = g.school_fit(TOPIK_U, PROFILE, TODAY)
    assert fit["GMAT/GRE"] == "官网未提及"
    assert fit["工作经验"] == "报考资格中未要求工作经验"
    assert "R1 截止 2026-05-01，已截止（以最新简章为准）" in fit["截止状态"]
    assert "R2 截止 2026-11-13，未截止（距截止 14 天）" in fit["截止状态"]
    missing = g.school_fit(NODATA_U, PROFILE, TODAY)
    assert missing["GMAT/GRE"] == missing["工作经验"] == "官网未说明"
    assert missing["截止状态"] == "资料中未提供轮次日期"


# ---------------------------------------------------------------------------
# 回答格式（无密钥）
# ---------------------------------------------------------------------------

def test_answer_per_school_with_source_and_no_tiers():
    result = match()
    draft = result["draft"]
    para = block(draft, "TopikU")
    assert "  语言：符合（要求 TOPIK 5级以上，你的 TOPIK 6级）｜原文：TOPIK 5级以上（仅限外国人）" in para
    assert "  本科专业：未限制专业｜原文：4年制大学毕业（预定）者，未限制专业" in para
    assert "  GMAT/GRE：官网未提及" in para
    assert "  工作经验：报考资格中未要求工作经验" in para
    assert "  出处：https://example.com/topik（数据整理日期 2026-10-29）" in para
    for removed in ("冲刺", "匹配", "保底", "综合", "分档", "申请者评估"):
        assert removed not in draft
    assert "学校记录未提供 GPA 要求，未作判断" in draft
    assert result["trace"] == ["router", "profile_extractor", "school_matcher"]


def test_state_fields():
    result = match()
    assert result["matched_schools"]["TopikU"]["语言"].startswith("符合")
    assert [d["school"] for d in result["docs"]] == ["TopikU", "MixU", "EngU", "NoDataU"]
    assert result["sources"][0] == "https://example.com/topik"


# ---------------------------------------------------------------------------
# 模型补充说明
# ---------------------------------------------------------------------------

GOOD = R(TopikU="语言要求 TOPIK 5级以上，你的 TOPIK 6级达到", MixU="与所修专业无关均可报考",
         EngU="可在 TOEFL、TOEIC、IELTS 中选其一", NoDataU="语言要求官网未说明")


def test_supplements_shown(monkeypatch):
    use_llm(monkeypatch, GOOD)
    result = match()
    assert "  补充说明：语言要求 TOPIK 5级以上，你的 TOPIK 6级达到" in block(result["draft"], "TopikU")
    assert result["trace"][-1] == "school_matcher:llm"


@pytest.mark.parametrize("reason", [
    "GPA 3.8 以上更有机会",            # 数字不在资料和画像中
    "截止日期为 2026-12-01",           # 日期不在资料中
    "不需要 GMAT",                     # 资料写的是"官网未提及"
    "很可能录取",                      # 断言录取概率
    "x" * (g.MAX_REASON_CHARS + 1),
    "   ",
])
def test_unsupported_supplement_not_shown_fit_unaffected(monkeypatch, reason):
    use_llm(monkeypatch, R(TopikU=reason, MixU="与所修专业无关均可报考"))
    result = match()
    para = block(result["draft"], "TopikU")
    assert "补充说明" not in para
    assert "  语言：符合（要求 TOPIK 5级以上，你的 TOPIK 6级）" in para
    assert "school_matcher:reason_dropped(TopikU)" in result["trace"]
    assert "补充说明：与所修专业无关均可报考" in block(result["draft"], "MixU")


def test_llm_request(monkeypatch):
    fake = use_llm(monkeypatch, GOOD)
    match()
    schema, method, (system, human) = fake.calls[0]
    assert schema is g.MatchReasons and method == "json_schema"
    for rule in ("具体分数以简章为准", "不得判断用户的英语成绩是否达标", "一律不写", "GPA 门槛",
                 "GMAT/GRE 对评审的影响", "不得断言录取概率", "符合或不符合", "系统判断的符合度",
                 "不要写截止状态", "不要陈述用户没有某项成绩", "不要判断某个分数算不算优秀"):
        assert rule in system.content
    payload = json.loads(human.content)
    first = payload["学校"][0]
    assert first["资料"]["school"] == "TopikU" and "aliases" not in first["资料"]
    assert first["系统判断的符合度"]["语言"].startswith("符合")
    assert "分档" not in human.content
    assert payload["申请者画像"]["language_score"] == "TOPIK 6级"


@pytest.mark.parametrize("response, entry", [
    (RuntimeError("down"), "school_matcher:llm_error"),
    (TimeoutError(), "school_matcher:llm_error"),
    (OutputParserException("bad json"), "school_matcher:llm_parse_error"),
    (None, "school_matcher:llm_parse_error"),
])
def test_llm_failure_no_supplements(monkeypatch, response, entry):
    use_llm(monkeypatch, response)
    result = match()
    assert result["trace"][-1] == entry
    assert "补充说明" not in result["draft"]
    assert "  语言：符合" in block(result["draft"], "TopikU")


def test_no_schools_no_llm_call(monkeypatch):
    fake = use_llm(monkeypatch, GOOD)
    result = match(config={"configurable": {"schools": []}})
    assert fake.calls == []
    assert result["docs"] == []


def test_llm_error_does_not_leak_message(monkeypatch):
    use_llm(monkeypatch, RuntimeError(f"invalid key {FAKE_KEY}"))
    assert FAKE_KEY not in json.dumps(match()["trace"])


def test_remove_supplements_only_named_schools():
    draft = g.format_match_answer(PROFILE, SCHOOLS, {s["school"]: g.school_fit(s, PROFILE, TODAY) for s in SCHOOLS},
                                  {"TopikU": "甲", "MixU": "乙"})
    kept = g.remove_supplements(draft, SCHOOLS, {"TopikU"})
    assert "补充说明" not in block(kept, "TopikU") and "补充说明：乙" in block(kept, "MixU")
    assert "补充说明" not in g.remove_supplements(draft, SCHOOLS, None)


# ---------------------------------------------------------------------------
# 真实学校数据：指定场景的符合度
# ---------------------------------------------------------------------------

def real_match(language_score):
    profile = {"gpa": "3.5/4.0", "language_score": language_score, "major": "经济学"}
    return g.school_matcher({"profile": profile, "trace": []})["matched_schools"]


def test_real_topik6_khu_fit():
    assert real_match("TOPIK 6级")["庆熙大学"]["语言"].startswith("符合（")


def test_real_topik3_khu_not_fit_ssu_fit():
    fits = real_match("TOPIK 3级")
    assert fits["庆熙大学"]["语言"].startswith("不符合（")
    assert fits["崇实大学"]["语言"].startswith("符合（要求 TOPIK 3级以上")


def test_real_english_score_unknown_where_no_specific_score():
    fits = real_match("TOEFL 95")
    assert fits["成均馆大学"]["语言"].startswith("无法判断，以简章为准")
    assert fits["崇实大学"]["语言"].startswith("无法判断，以简章为准")
    assert fits["仁济大学"]["语言"] == fits["国民大学"]["语言"] == "官网未说明"


def test_real_major_fit():
    fits = real_match("TOPIK 6级")
    assert {k: v["本科专业"] for k, v in fits.items()} == {
        "庆熙大学": "未限制专业", "仁济大学": "未限制专业", "成均馆大学": "未限制专业",
        "国民大学": "未限制专业", "崇实大学": "未限制专业",
    }


# ---------------------------------------------------------------------------
# 整图（真实学校数据）：补充说明由 verify 核对，未通过只删除该校补充说明
# ---------------------------------------------------------------------------

Q = "我本科经济学，GPA 3.5/4.0，TOPIK 6级，帮我选校"
REAL_SUPPLEMENTS = R(**{
    "庆熙大学": "语言要求 TOPIK 5级以上，你的 TOPIK 6级达到",
    "仁济大学": "与所修专业无关均可报考",
    "成均馆大学": "官网材料清单未列出",
    "国民大学": "未限制专业",
    "崇实大学": "外国人特别录取 TOPIK 3级以上，你的 TOPIK 6级达到",
})
EV_OK = g.VerifyResult(checks=[
    g.ClaimCheck(index=1, evidence=["language: TOPIK 5级以上（仅限外国人）", "TOPIK 6级"], reason="比较正确"),
    g.ClaimCheck(index=2, evidence=["TOPIK 3级以上", 'language_score": "TOPIK 6级'], reason="比较正确"),
])
EV_KHU_MISSING = g.VerifyResult(checks=[
    g.ClaimCheck(index=1, evidence=[], reason="无依据"),
    g.ClaimCheck(index=2, evidence=["TOPIK 3级以上", "TOPIK 6级"], reason="比较正确"),
])


def use_graph_llm(monkeypatch, verify_response):
    fake = FakeLLM({
        g.RouterResult: g.RouterResult(intent="profile"),
        g.ProfileResult: g.ProfileResult(gpa="3.5/4.0", language_score="TOPIK 6级", gmat_gre=None, major="经济学"),
        g.MatchReasons: REAL_SUPPLEMENTS,
        g.VerifyResult: verify_response,
    })
    monkeypatch.setattr(g, "get_llm", lambda: fake)
    return fake


def nodes(trace):
    return [t for t in trace if ":" not in t]


def test_graph_supplements_verified(monkeypatch):
    fake = use_graph_llm(monkeypatch, EV_OK)
    result = g.run(Q)
    assert result["verdict"] == "pass"
    assert nodes(result["trace"]) == ["router", "profile_extractor", "school_matcher", "verify"]
    assert result["trace"][-1] == "verify:llm_pass"
    assert "补充说明：语言要求 TOPIK 5级以上，你的 TOPIK 6级达到" in result["draft"]
    statements = fake.calls_for(g.VerifyResult)[0][1].content.split("【待核对陈述】")[1]
    assert statements.strip().splitlines() == [
        "1. 庆熙大学：语言要求 TOPIK 5级以上，你的 TOPIK 6级达到",
        "2. 崇实大学：外国人特别录取 TOPIK 3级以上，你的 TOPIK 6级达到",
    ]
    assert len(fake.calls) == 4


def test_graph_unsupported_supplement_removed_without_rewrite(monkeypatch):
    fake = use_graph_llm(monkeypatch, EV_KHU_MISSING)
    result = g.run(Q)
    assert result["verdict"] == "pass"
    assert nodes(result["trace"]) == ["router", "profile_extractor", "school_matcher", "verify"]
    assert "verify:supplement_removed(庆熙大学)" in result["trace"]
    assert "补充说明" not in block(result["draft"], "庆熙大学")
    assert "  语言：符合（要求 TOPIK 5级以上，你的 TOPIK 6级）" in block(result["draft"], "庆熙大学")
    assert "补充说明：外国人特别录取" in block(result["draft"], "崇实大学")
    assert result["messages"][-1].content == result["draft"]
    assert len(fake.calls) == 4


def test_graph_verify_llm_error_removes_all_supplements(monkeypatch):
    use_graph_llm(monkeypatch, RuntimeError("down"))
    result = g.run(Q)
    assert result["verdict"] == "pass"
    assert result["trace"][-2:] == ["verify:llm_error", "verify:supplement_removed(all)"]
    assert "补充说明" not in result["draft"]
    assert "  语言：符合" in block(result["draft"], "庆熙大学")


def test_graph_code_check_failure_goes_to_fallback(monkeypatch):
    use_graph_llm(monkeypatch, EV_OK)
    monkeypatch.setattr(g, "unsupported_negations", lambda draft, docs: ["不需要"])
    result = g.run(Q)
    assert nodes(result["trace"]) == ["router", "profile_extractor", "school_matcher", "verify", "fallback"]


def test_graph_without_key_passes_by_code():
    result = g.run("我想选校，GPA 3.6，本科专业金融，雅思 7.0")
    assert result["verdict"] == "pass"
    assert "补充说明" not in result["draft"]
    assert all(s["school"] in result["draft"] for s in g.load_schools())
