"""eval/run_eval.py 的测试：判定逻辑与三种配置的端到端运行（只用假模型，不调用真实 API）。"""

import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import graph as g
from fake_llm import FakeLLM

ROOT = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-333"


@pytest.fixture(autouse=True)
def restore_get_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "get_llm", g.get_llm)  # run_item 会替换 g.get_llm，测试结束后恢复
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path / "results")


def item(**kw):
    base = {"id": "A99", "category": "qa", "input": "q", "gold": {"must_contain": [], "must_not_contain": [], "per_school": {}}}
    return {**base, **kw}


# ---------------------------------------------------------------------------
# 判定函数
# ---------------------------------------------------------------------------

def test_node_path_filters_internal_entries():
    trace = ["__start__", "router", "router:llm", "requirement_retriever", "verify", "verify:llm_pass", "END"]
    assert run_eval.node_path(trace) == ["router", "requirement_retriever", "verify"]


def test_judge_route_expected_path_any():
    it = item(expected_path_any=[["router", "clarify"], ["router", "profile_extractor", "clarify"]])
    assert run_eval.judge_route(it, ["router", "profile_extractor", "clarify"])["ok"] is True
    assert run_eval.judge_route(it, ["router"])["ok"] is False
    assert run_eval.judge_route(it, None)["ok"] is None


def test_judge_contains_groups_and_forbidden():
    gold = {"must_contain": [["2026-05-15", "5月15日"], ["已截止"]], "must_not_contain": ["很可能录取"]}
    assert run_eval.judge_contains("截止 5月15日，已截止", gold)["ok"] is True
    missing = run_eval.judge_contains("截止 5月15日", gold)
    assert missing["ok"] is False and missing["missing_groups"] == [["已截止"]]
    hit = run_eval.judge_contains("2026-05-15 已截止，很可能录取", gold)
    assert hit["ok"] is False and hit["forbidden_hits"] == ["很可能录取"]


@pytest.mark.parametrize("gold, actual, ok", [
    ("符合", "符合（要求 TOPIK 5级以上，你的 TOPIK 6级）", True),
    ("符合", "不符合（要求 TOPIK 5级以上，你的 TOPIK 3级）", False),
    ("不符合", "不符合（要求 TOPIK 5级以上，你的 TOPIK 3级）", True),
    ("官网未说明", "官网未说明", True),
    ("无法判断，以简章为准", "无法判断，以简章为准（记录未给出 TOEFL 的具体分数要求）", True),
    ("无法判断，以简章为准", "无法判断（要求 TOPIK 5级以上，未提供 TOPIK 成绩）", False),
    ("不得判为符合", "无法判断（要求 TOPIK 5级以上，未提供 TOPIK 成绩）", True),
    ("不得判为符合", "符合（要求 TOPIK 5级以上，你的 TOPIK 6级）", False),
    ("与专业无关均可报考", "未限制专业", None),
])
def test_fit_matches(gold, actual, ok):
    assert run_eval.fit_matches(gold, actual) is ok


def real_match_answer(language_score):
    profile = {"gpa": "3.5/4.0", "language_score": language_score, "major": "经济学"}
    return g.school_matcher({"profile": profile, "trace": []})["draft"]


def test_fit_judgment_reads_school_line():
    answer = real_match_answer("TOPIK 3级")
    assert run_eval.fit_judgment(answer, "庆熙大学", "语言") == "不符合（要求 TOPIK 5级以上，你的 TOPIK 3级）"
    assert run_eval.fit_judgment(answer, "仁济大学", "本科专业") == "未限制专业"
    assert run_eval.fit_judgment(answer, "首尔大学", "语言") is None


def test_judge_fits_auto_and_manual():
    it = item(id="B99", gold={"must_contain": [], "must_not_contain": [], "per_school": {
        "庆熙大学": {"language": "不符合"},
        "崇实大学": {"language": "符合"},
        "仁济大学": {"major": "与专业无关均可报考"},
    }})
    result = run_eval.judge_fits(it, real_match_answer("TOPIK 3级"), "full")
    assert [(c["school"], c["ok"]) for c in result["checks"]] == [("庆熙大学", True), ("崇实大学", True)]
    assert [(m["school"], m["field"]) for m in result["manual"]] == [("仁济大学", "major")]


def test_judge_fits_missing_line():
    it = item(id="B99", gold={"must_contain": [], "must_not_contain": [], "per_school": {"庆熙大学": {"language": "符合"}}})
    full = run_eval.judge_fits(it, "抱歉，暂时无法确认。", "full")
    assert full["checks"][0]["ok"] is False and full["manual"] == []
    b0 = run_eval.judge_fits(it, "庆熙大学要求 TOPIK 5级，你符合。", "b0")
    assert b0["checks"] == [] and b0["manual"][0]["reason"].startswith("基线回答")


def test_load_items_skips_unchecked_by_default():
    all_items = json.loads(run_eval.TESTSET_PATH.read_text(encoding="utf-8"))["items"]
    unchecked = sum(bool(i.get("needs_check")) for i in all_items)
    assert len(run_eval.load_items(False, None)) == len(all_items) - unchecked
    assert len(run_eval.load_items(True, None)) == len(all_items)
    assert [i["id"] for i in run_eval.load_items(False, 3)] == ["A01", "A02", "A03"]
    assert all(not i.get("needs_check") for i in run_eval.load_items(False, None))


def test_counting_llm_with_max_tokens_shares_count():
    fake = FakeLLM({g.RouterResult: g.RouterResult(intent="qa")})
    counter = run_eval.CountingLLM(fake)
    bigger = counter.with_max_tokens(2048)
    bigger.with_structured_output(g.RouterResult).invoke([])
    counter.with_structured_output(g.RouterResult).invoke([])
    assert counter.calls == bigger.calls == 2


def test_compute_metrics():
    records = [
        {"id": "A01", "route": {"ok": True}, "fact_ok": True, "contains": {"forbidden_hits": []},
         "latency_s": 1.0, "model_calls": 2, "error": None},
        {"id": "F01", "route": {"ok": False}, "contains": {"forbidden_hits": ["2026-"]},
         "latency_s": 3.0, "model_calls": 0, "error": "RuntimeError"},
        {"id": "B01", "route": {"ok": None}, "contains": {"forbidden_hits": []}, "latency_s": 2.0, "model_calls": 4,
         "error": None, "fit": {"checks": [{"ok": True}, {"ok": False}], "manual": [{}]}},
    ]
    m = run_eval.compute_metrics(records)
    assert m["routing_accuracy"] == 0.5
    assert m["fact_accuracy"] == 1.0
    assert m["hallucination_rate"] == pytest.approx(1 / 3)
    assert m["hallucination_rate_F"] == 1.0
    assert m["fit_accuracy"] == 0.5 and m["fit_auto_checked"] == 2 and m["fit_manual_pending"] == 1
    assert m["avg_latency_s"] == 2.0 and m["avg_model_calls"] == 2.0
    assert m["error_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 端到端（无密钥时 full / no_verify 为规则版；模型由假模型代替）
# ---------------------------------------------------------------------------

def load_result(tmp_path):
    files = list((tmp_path / "results").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_full_rule_mode_end_to_end(tmp_path, capsys):
    assert run_eval.main(["--config", "full", "--limit", "3"]) == 0
    out = load_result(tmp_path)
    assert out["config"] == "full" and out["model"] == "规则版（未配置密钥）"
    assert out["item_ids"] == ["A01", "A02", "A03"]
    run = out["runs"][0]
    assert run["metrics"]["n_items"] == 3 and run["metrics"]["avg_model_calls"] == 0
    first = run["items"][0]
    assert first["path"] == ["router", "requirement_retriever", "verify"]
    assert first["route"]["ok"] is True
    assert first["eval_date"] == out["eval_date"]  # date_sensitive 题目记录评测日期
    assert "路由准确率" in capsys.readouterr().out


def test_repeat_and_average(tmp_path):
    assert run_eval.main(["--config", "full", "--limit", "2", "--repeat", "2"]) == 0
    out = load_result(tmp_path)
    assert [r["run"] for r in out["runs"]] == [1, 2]
    assert out["average"]["routing_accuracy"] == out["runs"][0]["metrics"]["routing_accuracy"]


def test_no_verify_skips_verify(tmp_path):
    assert run_eval.main(["--config", "no_verify", "--limit", "1"]) == 0
    rec = load_result(tmp_path)["runs"][0]["items"][0]
    assert rec["trace"][-2:] == ["verify", "verify:skipped"]
    assert rec["path"] == ["router", "requirement_retriever", "verify"]
    assert g.verify is not run_eval.skip_verify  # 构建后已恢复原 verify


def test_full_with_fake_model_counts_calls(monkeypatch, tmp_path):
    fake = FakeLLM({
        g.RouterResult: g.RouterResult(intent="qa"),
        g.VerifyResult: g.VerifyResult(checks=[g.ClaimCheck(index=1, evidence=["官网未提及"], reason="一致")]),
    })
    monkeypatch.setattr(run_eval, "get_llm", lambda: fake)
    assert run_eval.main(["--config", "full", "--limit", "1"]) == 0
    out = load_result(tmp_path)
    rec = out["runs"][0]["items"][0]
    # A01 问庆熙大学：GMAT/GRE 记录为"官网未提及"，回答改写为"资料中未提及……"，verify 调用模型核对
    assert rec["model_calls"] == 2
    assert rec["trace"] == ["router", "router:llm", "requirement_retriever", "verify", "verify:llm_pass"]
    assert out["model"] == run_eval.MODEL_NAME


class FakeChat:
    def __init__(self, text):
        self.text = text

    def invoke(self, messages):
        return AIMessage(content=self.text)


def test_b0_with_fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "get_llm", lambda: FakeChat("截止日期是 2026-05-15，已截止。"))
    assert run_eval.main(["--config", "b0", "--limit", "1"]) == 0
    out = load_result(tmp_path)
    rec = out["runs"][0]["items"][0]
    assert rec["model_calls"] == 1 and rec["path"] is None
    assert rec["route"]["ok"] is None
    assert rec["fact_ok"] is True
    assert rec["declined"] == {"ok": False, "hits": []}
    assert out["runs"][0]["metrics"]["routing_accuracy"] is None
    assert out["runs"][0]["metrics"]["b0_declined"] == 0


def test_b0_without_key_exits(tmp_path):
    assert run_eval.main(["--config", "b0", "--limit", "1"]) == 2
    assert not (tmp_path / "results").exists()


def test_exception_recorded_without_message(monkeypatch, tmp_path):
    class Broken:
        def stream(self, *a, **k):
            raise RuntimeError(f"boom {FAKE_KEY}")

    monkeypatch.setattr(run_eval, "build_app", lambda config: Broken())
    assert run_eval.main(["--config", "full", "--limit", "2"]) == 0
    raw = next((tmp_path / "results").glob("*.json")).read_text(encoding="utf-8")
    out = json.loads(raw)
    assert out["runs"][0]["metrics"]["error_rate"] == 1.0
    assert out["runs"][0]["items"][0]["error"] == "RuntimeError"
    assert FAKE_KEY not in raw


# ---------------------------------------------------------------------------
# 附加记录：补充说明删除、b0 表示无法回答
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trace_tail, final_count, removed, reason", [
    (["verify:llm_pass"], 5, 0, None),
    (["verify:llm_unsupported_claims(2)", "verify:unsupported:x｜未给出依据", "verify:supplement_removed(仁济大学)",
      "verify:supplement_removed(国民大学)"], 3, 2, "无依据"),
    (["verify:llm_parse_error", "verify:supplement_removed(all)"], 0, 5, "解析失败"),
    (["verify:llm_error", "verify:supplement_removed(all)"], 0, 5, "其他"),
])
def test_supplement_removal(trace_tail, final_count, removed, reason):
    trace = ["router", "profile_extractor", "school_matcher", "verify"] + trace_tail
    out = run_eval.supplement_removal(trace, 5, "补充说明：x\n" * final_count)
    assert out["generated"] == 5 and out["removed"] == removed
    assert sum(out["by_reason"].values()) == removed
    if reason:
        assert out["by_reason"][reason] == removed


def test_supplement_removal_not_school_matcher():
    assert run_eval.supplement_removal(["router", "requirement_retriever", "verify"], None, "") is None


def test_full_records_supplements_end_to_end(monkeypatch, tmp_path):
    ok = g.SchoolReason(school="崇实大学", reason="语言要求 TOPIK 3级以上，你的 TOPIK 4级")
    fake = FakeLLM({
        g.RouterResult: g.RouterResult(intent="profile"),
        g.ProfileResult: g.ProfileResult(gpa="3.2/4.0", language_score="TOPIK 4级", gmat_gre=None, major="计算机"),
        g.MatchReasons: g.MatchReasons(reasons=[ok]),
        g.VerifyResult: g.VerifyResult(checks=[g.ClaimCheck(index=1, evidence=[], reason="无")]),
    })
    monkeypatch.setattr(run_eval, "get_llm", lambda: fake)
    monkeypatch.setattr(run_eval, "load_items", lambda *a: [item(
        id="B99", category="profile", input="我本科是计算机专业，GPA 3.2/4.0，TOPIK 4级，帮我选校",
        gold={"must_contain": [], "must_not_contain": [], "per_school": {}})])
    assert run_eval.main(["--config", "full"]) == 0
    out = load_result(tmp_path)
    rec = out["runs"][0]["items"][0]
    assert rec["supplements"] == {"generated": 1, "removed": 1, "by_reason": {"无依据": 1, "解析失败": 0, "其他": 0}}
    m = out["runs"][0]["metrics"]
    assert m["supplement_removal_answers"] == 1 and m["supplement_removal_rate"] == 1.0
    assert m["removed_no_evidence"] == 1 and m["b0_declined"] is None


def test_b0_declined_counted(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "get_llm", lambda: FakeChat("我没有实时数据，无法确定具体截止日期，建议查阅官网。"))
    assert run_eval.main(["--config", "b0", "--limit", "2"]) == 0
    out = load_result(tmp_path)
    assert out["runs"][0]["metrics"]["b0_declined"] == 2
    assert "无法确定" in out["runs"][0]["items"][0]["declined"]["hits"]


# ---------------------------------------------------------------------------
# 禁用词记录与 failures.md
# ---------------------------------------------------------------------------

def test_forbidden_hits_recorded_and_listed_in_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "load_items", lambda *a: [item(
        id="A98", input="庆熙大学经营大学院一般硕士第一次的报名截止日期是哪天？",
        expected_path=["router", "requirement_retriever", "verify"],
        gold={"must_contain": [], "must_not_contain": ["庆熙", "首尔大学"], "per_school": {}})])
    assert run_eval.main(["--config", "full"]) == 0
    rec = load_result(tmp_path)["runs"][0]["items"][0]
    assert rec["forbidden_hits"] == ["庆熙"]

    assert run_eval.main(["--report"]) == 0
    failures = (tmp_path / "results" / "failures.md").read_text(encoding="utf-8")
    assert "初步分类，需要人工确认" in failures
    section = failures[failures.index("## 触发禁用词的题"):]
    assert "full · A98" in section and "触发的禁用词：庆熙" in section
    assert (tmp_path / "results" / "manual_review.md").exists()


def test_report_without_failures(monkeypatch, tmp_path):
    assert run_eval.main(["--config", "full", "--limit", "1"]) == 0
    assert run_eval.main(["--report"]) == 0
    failures = (tmp_path / "results" / "failures.md").read_text(encoding="utf-8")
    assert "无失败题目" in failures and failures.rstrip().endswith("无。")


def test_classify_route_and_retrieval():
    base = {"id": "A01", "error": None, "contains": {"missing_groups": [], "forbidden_hits": []},
            "expected_paths": [["router", "requirement_retriever", "verify"]]}
    verify_fail = {**base, "route": {"ok": False}, "path": ["router", "requirement_retriever", "verify", "fallback"],
                   "trace": ["verify", "verify:llm_unsupported_claims(1)"]}
    assert run_eval.classify(verify_fail) == ["verify 误判"]
    routed = {**base, "route": {"ok": False}, "path": ["router", "fallback"], "trace": []}
    assert run_eval.classify(routed) == ["路由错误"]
    missing = {**base, "route": {"ok": True}, "path": [], "trace": [],
               "contains": {"missing_groups": [["已截止"]], "forbidden_hits": []}}
    assert run_eval.classify(missing) == ["检索遗漏"]


def test_classify_clarify_without_profile_and_b0():
    base = {"id": "G02", "error": None, "contains": {"missing_groups": [], "forbidden_hits": []},
            "expected_paths": [["router"]], "trace": []}
    assert run_eval.classify({**base, "route": {"ok": False}, "path": ["router", "clarify"]}) == ["路由错误"]
    profile = {**base, "route": {"ok": False}, "path": ["router", "profile_extractor", "clarify"]}
    assert run_eval.classify(profile) == ["画像抽取错误"]
    b0 = {**base, "id": "A01", "route": {"ok": None}, "path": None,
          "contains": {"missing_groups": [["已截止"]], "forbidden_hits": []}}
    assert run_eval.classify(b0) == ["其他"]
