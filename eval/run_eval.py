"""评测脚本：在 eval/testset.json 上评测三种配置。

用法：
    .venv/bin/python eval/run_eval.py --config full
    .venv/bin/python eval/run_eval.py --config b0 --repeat 3
    .venv/bin/python eval/run_eval.py --config no_verify --limit 5 --include-unchecked
    .venv/bin/python eval/run_eval.py --config b1 --repeat 3
    .venv/bin/python eval/run_eval.py --report   # 读取各配置最新结果，生成 failures.md、manual_review.md、removed_statements.md

配置：
    full       完整系统（LangGraph 工作流）
    b0         基线：同一个模型直接回答，不使用 schools.json，不走图
    no_verify  完整系统，但跳过 verify，直接把草稿作为最终回答
    b1         基线：把 schools.json 全部学校记录放进提示词，同一个模型直接回答，不走图、不路由、不校验

指标（每次运行都计算，最后输出每次的值和平均值）：
    路由准确率      实际节点路径（去掉带冒号的内部条目和 START/END）与 expected_path 或 expected_path_any 之一一致；b0、b1 不计
    事实准确率      A、C 类：每组 must_contain 至少出现一个词，且不触发 must_not_contain
    幻觉率          所有题目中触发 must_not_contain 的比例；F 类单独统计
    符合度判断正确率 B 类：per_school 中能自动判定的项与回答里该校的判断一致的比例；不能自动判定的写入结果文件供人工检查
    平均延迟、平均模型调用次数、异常率

结果写入 eval/results/<配置>_<时间>.json，每题记录触发的禁用词（forbidden_hits）。需要真实模型时读取 ANTHROPIC_API_KEY（环境变量、.env 或 st.secrets）；
脚本只显示是否配置了密钥，不输出密钥本身。
"""

import argparse
import copy
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

import graph as g  # noqa: E402
from config import api_key_hint  # noqa: E402
from llm import MODEL_NAME, get_llm, with_max_tokens  # noqa: E402

TESTSET_PATH = ROOT / "eval" / "testset.json"
RESULTS_DIR = ROOT / "eval" / "results"
CONFIGS = ("full", "b0", "no_verify", "b1")
BASELINES = ("b0", "b1")        # 不走图的基线：路由不计，符合度无法自动判定时交人工检查

FACT_PREFIXES = ("A", "C")      # 事实准确率统计的题目类别（按 id 首字母）
FIT_PREFIX = "B"                # 符合度判断
NO_ANSWER_PREFIX = "F"          # 幻觉率单独统计
NON_NODES = {"START", "END", "__start__", "__end__"}

# per_school 字段 → 回答中的行标签
FIT_LABELS = {"language": "语言", "major": "本科专业", "gmat_gre": "GMAT/GRE"}
# 能自动判定的符合度类别；按长度从长到短匹配，避免"符合"误匹配"不符合"、"无法判断"误匹配"无法判断，以简章为准"
FIT_CATEGORIES = sorted({g.FIT_YES, g.FIT_NO, g.FIT_UNKNOWN, g.FIT_UNSTATED, g.FIT_BY_BROCHURE}, key=len, reverse=True)
NEGATED_PREFIX = "不得判为"

B0_SYSTEM_PROMPT = "你是经营学硕士留学申请顾问。请用中文直接回答用户的问题。"
B1_SYSTEM_PROMPT = (
    "你是经营学硕士留学申请顾问。请用中文回答用户的问题。"
    "依据给出的资料回答，资料里没有的内容说明资料中未提及。\n\n【资料】\n{schools}"
)


def baseline_prompt(config: str) -> str:
    if config == "b1":
        return B1_SYSTEM_PROMPT.format(schools=json.dumps(g.load_schools(), ensure_ascii=False, indent=1))
    return B0_SYSTEM_PROMPT

# b0 回答"表示无法回答"的关键词（简单匹配，命中的词写入结果文件供人工复核）
DECLINE_PHRASES = (
    "无法确定", "无法确认", "无法提供", "无法回答", "无法给出", "无法保证", "不确定", "不清楚",
    "没有实时", "无法获取", "无法访问", "无法查询", "建议查阅", "建议直接查阅", "请查阅", "以官网为准",
)

# verify 删除补充说明的原因分类（取自 trace）
REMOVAL_NO_EVIDENCE, REMOVAL_PARSE_ERROR, REMOVAL_OTHER = "无依据", "解析失败", "其他"


# ---------------------------------------------------------------------------
# 模型调用计数
# ---------------------------------------------------------------------------

class CountingLLM:
    """包装模型，统计 invoke 次数（含 with_structured_output(...).invoke）。"""

    def __init__(self, llm: Any):
        self._llm = llm
        self._count = [0]  # 副本共用

    @property
    def calls(self):
        return self._count[0]

    def with_structured_output(self, schema, **kwargs):
        bound = self._llm.with_structured_output(schema, **kwargs)
        counter = self

        class _Counted:
            def invoke(self, messages):
                counter._count[0] += 1
                return bound.invoke(messages)

        return _Counted()

    def with_max_tokens(self, max_tokens):
        """verify 用更大的 max_tokens；返回的副本与原对象共用计数。"""
        clone = copy.copy(self)
        clone._llm = with_max_tokens(self._llm, max_tokens)
        return clone

    def invoke(self, messages):
        self._count[0] += 1
        return self._llm.invoke(messages)


# ---------------------------------------------------------------------------
# 判定函数
# ---------------------------------------------------------------------------

def node_path(trace: list[str]) -> list[str]:
    return [t for t in trace if ":" not in t and t not in NON_NODES]


def expected_paths(item: dict) -> list[list[str]]:
    if item.get("expected_path"):
        return [item["expected_path"]]
    return item.get("expected_path_any") or []


def judge_route(item: dict, path: Optional[list[str]]) -> dict:
    expected = expected_paths(item)
    if path is None or not expected:
        return {"ok": None, "reason": "不计" if path is None else "题目未给出预期路径"}
    ok = path in expected
    return {"ok": ok, "reason": "一致" if ok else f"实际 {path}，预期 {expected}"}


def judge_contains(answer: str, gold: dict) -> dict:
    missing = [group for group in gold.get("must_contain", []) if not any(w in answer for w in group)]
    hits = [w for w in gold.get("must_not_contain", []) if w in answer]
    return {"ok": not missing and not hits, "missing_groups": missing, "forbidden_hits": hits}


def fit_judgment(answer: str, school: str, label: str) -> Optional[str]:
    """回答中某校"<标签>：<判断>"行的判断部分（"｜原文"之前）；找不到返回 None。"""
    for para in answer.split("\n\n"):
        if not para.strip().startswith(f"{school} "):
            continue
        for line in para.splitlines():
            line = line.strip()
            if line.startswith(f"{label}："):
                return line[len(label) + 1:].split("｜", 1)[0].strip()
    return None


def fit_category(text: str) -> Optional[str]:
    return next((c for c in FIT_CATEGORIES if text.startswith(c)), None)


def fit_matches(gold_value: str, actual: str) -> Optional[bool]:
    """自动判定：gold 为四类结果之一时比较类别；gold 为"不得判为X"时检查实际不是 X。
    其他写法（如"与专业无关均可报考""官网未提及，不得说有加分"）返回 None，交给人工检查。"""
    actual_cat = fit_category(actual)
    if gold_value in FIT_CATEGORIES:
        return actual_cat == gold_value
    if gold_value.startswith(NEGATED_PREFIX) and gold_value[len(NEGATED_PREFIX):] in FIT_CATEGORIES:
        return actual_cat != gold_value[len(NEGATED_PREFIX):]
    return None


def judge_fits(item: dict, answer: str, config: str) -> dict:
    """返回 {"checks": [...自动判定...], "manual": [...需人工检查...]}。"""
    checks, manual = [], []
    for school, fields in item["gold"].get("per_school", {}).items():
        for field, gold_value in fields.items():
            label = FIT_LABELS.get(field)
            actual = fit_judgment(answer, school, label) if label else None
            entry = {"school": school, "field": field, "gold": gold_value, "actual": actual}
            auto = gold_value in FIT_CATEGORIES or (
                gold_value.startswith(NEGATED_PREFIX) and gold_value[len(NEGATED_PREFIX):] in FIT_CATEGORIES
            )
            if not auto:
                manual.append({**entry, "reason": "标准判断不是四类结果之一，需人工检查"})
            elif actual is None and config in BASELINES:
                manual.append({**entry, "reason": "基线回答没有结构化的逐校判断，需人工检查"})
            elif actual is None:
                checks.append({**entry, "ok": False, "reason": "回答中未找到该校的判断"})
            else:
                ok = fit_matches(gold_value, actual)
                checks.append({**entry, "ok": ok, "reason": "一致" if ok else "不一致"})
    return {"checks": checks, "manual": manual}


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

def load_items(include_unchecked: bool, limit: Optional[int]) -> list[dict]:
    items = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))["items"]
    if not include_unchecked:
        items = [i for i in items if not i.get("needs_check")]
    return items[:limit] if limit else items


def skip_verify(state: g.AdvisorState) -> dict:
    """no_verify 配置：不做任何核对，直接把草稿作为最终回答。"""
    draft = state.get("draft", "")
    return {
        "verdict": "pass",
        "trace": g._trace(state, "verify") + ["verify:skipped"],
        "messages": [AIMessage(content=draft)],
    }


def build_app(config: str):
    if config in BASELINES:
        return None
    if config == "no_verify":
        original = g.verify
        g.verify = skip_verify  # build_graph 在构建时读取 g.verify
        try:
            return g.build_graph()
        finally:
            g.verify = original
    return g.build_graph()


def final_answer(result: dict) -> str:
    ai = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    return ai[-1].content if ai else result.get("draft", "")


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 内容块列表
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def supplement_texts(draft: str) -> dict[str, str]:
    """选校回答中各校的补充说明原文 {学校: 补充说明}。"""
    names = [r["school"] for r in g.load_schools()]
    out = {}
    for para in draft.split("\n\n"):
        lines = para.strip().splitlines()
        school = next((n for n in names if lines and lines[0].startswith(f"{n} ")), None)
        if school is None:
            continue
        for line in lines:
            if line.strip().startswith(g.SUPPLEMENT_LABEL):
                out[school] = line.strip()[len(g.SUPPLEMENT_LABEL):]
    return out


def invoke_graph(app: Any, question: str) -> tuple[dict, Optional[dict[str, str]]]:
    """运行图，返回 (最终状态, school_matcher 生成的各校补充说明原文；未经过 school_matcher 为 None)。"""
    final: dict = {}
    generated = None
    for mode, chunk in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        elif "school_matcher" in chunk:
            generated = supplement_texts(chunk["school_matcher"].get("draft") or "")
    return final, generated


def supplement_removal(trace: list[str], generated: Optional[int], final_draft: str) -> Optional[dict]:
    """verify 删除补充说明的条数及原因分类（无依据 / 解析失败 / 其他）。"""
    if generated is None:
        return None
    removed_entries = [t for t in trace if t.startswith("verify:supplement_removed(")]
    removed = generated - final_draft.count(g.SUPPLEMENT_LABEL) if removed_entries else 0
    by_reason = {REMOVAL_NO_EVIDENCE: 0, REMOVAL_PARSE_ERROR: 0, REMOVAL_OTHER: 0}
    if removed:
        if "verify:llm_parse_error" in trace:
            by_reason[REMOVAL_PARSE_ERROR] = removed
        elif any(t.startswith("verify:llm_unsupported_claims(") for t in trace):
            by_reason[REMOVAL_NO_EVIDENCE] = removed
        else:
            by_reason[REMOVAL_OTHER] = removed  # 如 verify:llm_error
    return {"generated": generated, "removed": removed, "by_reason": by_reason}


def run_item(config: str, item: dict, app: Any, real_llm: Any, today: str) -> dict:
    counter = CountingLLM(real_llm) if real_llm is not None else None
    g.get_llm = lambda: counter
    record: dict[str, Any] = {"id": item["id"], "category": item["category"], "input": item["input"]}
    if item.get("date_sensitive"):
        record["eval_date"] = today

    start = time.perf_counter()
    trace: Optional[list[str]] = None
    try:
        if config in BASELINES:
            response = counter.invoke([SystemMessage(content=baseline_prompt(config)), HumanMessage(content=item["input"])])
            answer = text_of(response.content)
        else:
            result, generated = invoke_graph(app, item["input"])
            answer = final_answer(result)
            trace = result.get("trace", [])
            count = len(generated) if generated is not None else None
            record["supplements"] = supplement_removal(trace, count, result.get("draft", ""))
            if generated is not None:
                record["supplement_texts"] = generated  # school_matcher 生成的原文（verify 之前）
            record["profile"] = result.get("profile") or {}
        record["error"] = None
    except Exception as e:  # 只记录异常类型，异常内容可能包含请求细节
        answer, record["error"] = "", type(e).__name__
        trace = [] if config not in BASELINES else None
    record["latency_s"] = round(time.perf_counter() - start, 3)
    record["model_calls"] = counter.calls if counter else 0
    record["answer"] = answer
    record["trace"] = trace
    record["path"] = node_path(trace) if trace is not None else None

    if config in BASELINES:
        hits = [p for p in DECLINE_PHRASES if p in answer]
        record["declined"] = {"ok": bool(hits), "hits": hits}

    prefix = item["id"][0]
    record["route"] = judge_route(item, record["path"]) if config not in BASELINES else {"ok": None, "reason": f"{config} 不计"}
    record["contains"] = judge_contains(answer, item["gold"])
    record["forbidden_hits"] = record["contains"]["forbidden_hits"]
    record["expected_paths"] = expected_paths(item)
    if prefix in FACT_PREFIXES:
        record["fact_ok"] = record["contains"]["ok"]
    if prefix == FIT_PREFIX:
        record["fit"] = judge_fits(item, answer, config)
    return record


def ratio(values: list[bool]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_metrics(records: list[dict]) -> dict:
    routes = [r["route"]["ok"] for r in records if r["route"]["ok"] is not None]
    facts = [r["fact_ok"] for r in records if "fact_ok" in r]
    halluc = [bool(r["contains"]["forbidden_hits"]) for r in records]
    halluc_f = [bool(r["contains"]["forbidden_hits"]) for r in records if r["id"].startswith(NO_ANSWER_PREFIX)]
    fits = [c["ok"] for r in records if "fit" in r for c in r["fit"]["checks"]]
    sups = [r["supplements"] for r in records if r.get("supplements")]
    generated = sum(x["generated"] for x in sups)
    removed = sum(x["removed"] for x in sups)
    declined = [r for r in records if "declined" in r]
    return {
        "routing_accuracy": ratio(routes),
        "fact_accuracy": ratio(facts),
        "hallucination_rate": ratio(halluc),
        "hallucination_rate_F": ratio(halluc_f),
        "fit_accuracy": ratio(fits),
        "fit_auto_checked": len(fits),
        "fit_manual_pending": sum(len(r["fit"]["manual"]) for r in records if "fit" in r),
        "avg_latency_s": sum(r["latency_s"] for r in records) / len(records) if records else None,
        "avg_model_calls": sum(r["model_calls"] for r in records) / len(records) if records else None,
        "error_rate": ratio([r["error"] is not None for r in records]),
        "n_items": len(records),
        # 附加记录（不属于上面的指标）
        "supplement_answers": len(sups),
        "supplement_removal_answers": sum(x["removed"] > 0 for x in sups),
        "supplements_generated": generated,
        "supplements_removed": removed,
        "supplement_removal_rate": removed / generated if generated else None,
        "removed_no_evidence": sum(x["by_reason"][REMOVAL_NO_EVIDENCE] for x in sups),
        "removed_parse_error": sum(x["by_reason"][REMOVAL_PARSE_ERROR] for x in sups),
        "removed_other": sum(x["by_reason"][REMOVAL_OTHER] for x in sups),
        "baseline_declined": sum(r["declined"]["ok"] for r in declined) if declined else None,
    }


def average(runs: list[dict]) -> dict:
    keys = runs[0]["metrics"].keys() if runs else []
    out = {}
    for k in keys:
        vals = [r["metrics"][k] for r in runs if r["metrics"][k] is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


METRIC_ROWS = [
    ("routing_accuracy", "路由准确率", "pct"),
    ("fact_accuracy", "事实准确率（A、C）", "pct"),
    ("hallucination_rate", "幻觉率（全部）", "pct"),
    ("hallucination_rate_F", "幻觉率（F 类）", "pct"),
    ("fit_accuracy", "符合度判断正确率（B）", "pct"),
    ("fit_auto_checked", "  自动判定项数", "num"),
    ("fit_manual_pending", "  待人工检查项数", "num"),
    ("avg_latency_s", "平均延迟（秒）", "num"),
    ("avg_model_calls", "平均模型调用次数", "num"),
    ("error_rate", "异常率", "pct"),
    ("n_items", "题目数", "num"),
    ("supplement_removal_answers", "删除补充说明的回答数", "num"),
    ("supplements_removed", "  删除的补充说明条数", "num"),
    ("supplements_generated", "  生成的补充说明条数", "num"),
    ("supplement_removal_rate", "  删除比例", "pct"),
    ("removed_no_evidence", "  原因：无依据", "num"),
    ("removed_parse_error", "  原因：解析失败", "num"),
    ("removed_other", "  原因：其他", "num"),
    ("baseline_declined", "基线表示无法回答的题数", "num"),
]


def fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if kind == "pct" else f"{value:.2f}".rstrip("0").rstrip(".")


def print_summary(config: str, runs: list[dict], avg: dict) -> None:
    headers = ["指标"] + [f"第{i}次" for i in range(1, len(runs) + 1)] + ["平均"]
    rows = [
        [label] + [fmt(r["metrics"][key], kind) for r in runs] + [fmt(avg.get(key), kind)]
        for key, label, kind in METRIC_ROWS
    ]
    width = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    print(f"\n配置：{config}")
    for row in [headers] + rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, width)))


# ---------------------------------------------------------------------------
# 报告：failures.md、manual_review.md
# ---------------------------------------------------------------------------

VERIFY_FAIL_ENTRIES = (
    "verify:llm_unsupported_claims(", "verify:dates_not_in_sources", "verify:negation_not_in_sources",
    "verify:admission_claim", "verify:llm_error", "verify:llm_parse_error", "verify:no_sources", "verify:empty_draft",
)
CLASS_ROUTE, CLASS_RETRIEVAL, CLASS_PROFILE, CLASS_VERIFY, CLASS_OTHER = (
    "路由错误", "检索遗漏", "画像抽取错误", "verify 误判", "其他",
)


def failed_checks(rec: dict) -> list[str]:
    """该题未通过的判定项；空列表表示通过。"""
    out = []
    if rec.get("error"):
        out.append(f"运行异常（{rec['error']}）")
    if rec["route"]["ok"] is False:
        out.append("路由")
    missing = rec["contains"]["missing_groups"]
    if missing:
        out.append("缺少必含词：" + "；".join("/".join(grp) for grp in missing))
    if rec.get("forbidden_hits"):
        out.append("触发禁用词：" + "、".join(rec["forbidden_hits"]))
    for c in (rec.get("fit") or {}).get("checks", []):
        if c["ok"] is False:
            out.append(f"符合度 {c['school']}·{c['field']}：标准「{c['gold']}」，实际「{c['actual'] or '未找到'}」")
    return out


def classify(rec: dict) -> list[str]:
    """初步分类（规则推断，需要人工确认）。"""
    if rec.get("error") or rec.get("path") is None:  # 运行异常；b0 不走图，无路由、检索、画像环节
        return [CLASS_OTHER]
    trace = rec.get("trace") or []
    path = rec.get("path") or []
    classes = []
    if rec["route"]["ok"] is False:
        verify_failed = any(t.startswith(VERIFY_FAIL_ENTRIES) for t in trace)
        expected_has_clarify = any("clarify" in p for p in rec.get("expected_paths", []))
        if verify_failed and ("fallback" in path or path.count("verify") > 1):
            classes.append(CLASS_VERIFY)
        elif "profile_extractor" in path and "clarify" in path and not expected_has_clarify:
            classes.append(CLASS_PROFILE)
        else:
            classes.append(CLASS_ROUTE)
    elif rec["contains"]["missing_groups"] and rec["id"][0] in FACT_PREFIXES:
        classes.append(CLASS_RETRIEVAL)
    if any(c["ok"] is False for c in (rec.get("fit") or {}).get("checks", [])) and not classes:
        classes.append(CLASS_PROFILE if not (rec.get("profile") or {}).get("language_score") else CLASS_OTHER)
    if not classes:
        classes.append(CLASS_OTHER)
    return classes


def latest_results(results_dir: Path) -> dict[str, dict]:
    out = {}
    for config in CONFIGS:
        files = sorted(results_dir.glob(f"{config}_*.json"))
        if files:
            out[config] = {"file": files[-1].name, **json.loads(files[-1].read_text(encoding="utf-8"))}
    return out


def md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_report(results_dir: Path) -> list[Path]:
    results = latest_results(results_dir)
    fail_lines = [
        "# 评测失败题目",
        "",
        "> 分类是根据路径和 trace 用规则推断的**初步分类，需要人工确认**。",
        "",
    ]
    forbidden_lines = ["## 触发禁用词的题", "", "需人工判断是真幻觉还是误触发。", ""]
    manual_lines = ["# 需要人工检查的清单", ""]
    fit_manual, baseline_answers = [], []
    for config, data in results.items():
        fail_lines += [f"## {config}（{data['file']}）", ""]
        any_fail = False
        for run in data["runs"]:
            for rec in run["items"]:
                checks = failed_checks(rec)
                if not checks:
                    continue
                any_fail = True
                expected = " 或 ".join(" → ".join(p) for p in rec.get("expected_paths", [])) or "—"
                actual = " → ".join(rec["path"]) if rec.get("path") is not None else "—（不走图）"
                fail_lines += [
                    f"### {rec['id']}（第 {run['run']} 次）",
                    f"- 输入：{rec['input']}",
                    f"- 期望路径：{expected}",
                    f"- 实际路径：{actual}",
                    f"- 失败的判定项：{'；'.join(checks)}",
                    f"- 原因（路由判定）：{rec['route']['reason']}",
                    f"- 初步分类：{'、'.join(classify(rec))}（初步分类，需要人工确认）",
                    "",
                ]
                if rec.get("forbidden_hits"):
                    forbidden_lines += [
                        f"### {config} · {rec['id']}（第 {run['run']} 次）",
                        f"- 输入：{rec['input']}",
                        f"- 触发的禁用词：{'、'.join(rec['forbidden_hits'])}",
                        f"- 回答：{rec['answer']}",
                        "",
                    ]
            fit_manual += [(config, run["run"], m) for m in run.get("manual_fit_checks", [])]
            if config in BASELINES:
                baseline_answers += [(config, run["run"], rec) for rec in run["items"]]
        if not any_fail:
            fail_lines += ["无失败题目。", ""]
    if len(forbidden_lines) == 4:
        forbidden_lines.append("无。")

    manual_lines += ["## 1. 自动判定不了的符合度项", ""]
    if fit_manual:
        manual_lines += ["| 配置 | 次 | 题目 | 学校 | 字段 | 标准判断 | 回答中的判断 | 原因 |", "|---|---|---|---|---|---|---|---|"]
        manual_lines += [
            f"| {c} | {n} | {m['id']} | {m['school']} | {m['field']} | {md_cell(m['gold'])} | "
            f"{md_cell(m['actual'] or '—')} | {md_cell(m['reason'])} |"
            for c, n, m in fit_manual
        ]
    else:
        manual_lines.append("无。")
    manual_lines += ["", "## 2. 触发禁用词的题", "", "见 failures.md 中“触发禁用词的题”一节。", "", "## 3. 基线（b0、b1）的回答", ""]
    for config, n, rec in baseline_answers:
        declined = rec.get("declined") or {}
        manual_lines += [
            f"### {config} · {rec['id']}（第 {n} 次）",
            f"- 输入：{rec['input']}",
            f"- 关键词判断表示无法回答：{'是（' + '、'.join(declined.get('hits', [])) + '）' if declined.get('ok') else '否'}",
            f"- 回答：{rec['answer']}",
            "",
        ]
    if not baseline_answers:
        manual_lines.append("没有基线结果。")

    results_dir.mkdir(parents=True, exist_ok=True)
    failures_path, manual_path = results_dir / "failures.md", results_dir / "manual_review.md"
    removed_path = results_dir / "removed_statements.md"
    failures_path.write_text("\n".join(fail_lines + forbidden_lines) + "\n", encoding="utf-8")
    manual_path.write_text("\n".join(manual_lines) + "\n", encoding="utf-8")
    removed_path.write_text("\n".join(removed_statement_lines(results)) + "\n", encoding="utf-8")
    return [failures_path, manual_path, removed_path]


def removal_detail(trace: list[str], school: str) -> str:
    """某校补充说明被删的原因（取自 verify 的 trace 条目）。"""
    if "verify:llm_parse_error" in trace:
        return "解析失败（verify:llm_parse_error），全部补充说明被删除"
    if "verify:llm_error" in trace:
        return "其他（verify:llm_error），全部补充说明被删除"
    prefix = f"verify:unsupported:{school}："
    details = [t[len(prefix):].split("｜", 1)[-1] for t in trace if t.startswith(prefix)]
    return "无依据：" + "；".join(details) if details else "无依据（trace 中未找到该校的详情）"


def removed_statement_lines(results: dict[str, dict]) -> list[str]:
    lines = [
        "# verify 删除的补充说明",
        "",
        "full 配置每次运行中被 verify 删除的补充说明原文和删除原因，以及 no_verify 配置同一题、同一次运行中该校的补充说明原文。",
        "只列出，不判断对错。",
        "",
    ]
    full, no_verify = results.get("full"), results.get("no_verify")
    if full is None:
        return lines + ["没有 full 结果。"]
    nv_runs = {run["run"]: {r["id"]: r for r in run["items"]} for run in (no_verify or {}).get("runs", [])}
    found = False
    for run in full["runs"]:
        for rec in run["items"]:
            texts = rec.get("supplement_texts") or {}
            if not (rec.get("supplements") or {}).get("removed"):
                continue
            kept = supplement_texts(rec["answer"])
            nv_rec = nv_runs.get(run["run"], {}).get(rec["id"])
            for school, text in texts.items():
                if school in kept:
                    continue
                found = True
                nv_text = (nv_rec or {}).get("supplement_texts", {}).get(school) if nv_rec else None
                lines += [
                    f"## {rec['id']} · 第 {run['run']} 次 · {school}",
                    f"- 输入：{rec['input']}",
                    f"- full 被删的补充说明：{text}",
                    f"- 删除原因：{removal_detail(rec.get('trace') or [], school)}",
                    f"- no_verify 同题同次的补充说明：{nv_text if nv_text is not None else '（无）'}",
                    "",
                ]
    if not found:
        lines.append("full 配置中没有被删除的补充说明。")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="在 eval/testset.json 上评测")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--repeat", type=int, default=1, help="重复运行次数")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 K 题（在跳过 needs_check 之后计数）")
    parser.add_argument("--include-unchecked", action="store_true", help="包含 needs_check 为 true 的题目")
    parser.add_argument("--report", action="store_true", help="不运行评测，读取各配置最新结果生成 failures.md 等报告")
    args = parser.parse_args(argv)
    if args.report:
        for path in write_report(RESULTS_DIR):
            print(f"已写入：{path}")
        return 0
    if args.config is None:
        parser.error("需要 --config 或 --report")

    print(api_key_hint())
    real_llm = get_llm()
    if real_llm is None and args.config in BASELINES:
        print(f"{args.config} 配置需要模型，未检测到密钥，已退出。")
        return 2

    items = load_items(args.include_unchecked, args.limit)
    today = date.today().isoformat()
    app = build_app(args.config)
    original_get_llm = g.get_llm
    runs = []
    try:
        for n in range(1, args.repeat + 1):
            records = [run_item(args.config, item, app, real_llm, today) for item in items]
            runs.append({
                "run": n,
                "metrics": compute_metrics(records),
                "manual_fit_checks": [
                    {"id": r["id"], **m} for r in records if "fit" in r for m in r["fit"]["manual"]
                ],
                "items": records,
            })
    finally:
        g.get_llm = original_get_llm

    avg = average(runs)
    started = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.config}_{started}.json"
    out_path.write_text(json.dumps({
        "config": args.config,
        "model": MODEL_NAME if real_llm is not None else "规则版（未配置密钥）",
        "eval_date": today,
        "args": vars(args),
        "item_ids": [i["id"] for i in items],
        "average": avg,
        "runs": runs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print_summary(args.config, runs, avg)
    print(f"\n结果已写入：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
