"""评测脚本：在 eval/testset.json 上评测三种配置。

用法：
    .venv/bin/python eval/run_eval.py --config full
    .venv/bin/python eval/run_eval.py --config b0 --repeat 3
    .venv/bin/python eval/run_eval.py --config no_verify --limit 5 --include-unchecked

配置：
    full       完整系统（LangGraph 工作流）
    b0         基线：同一个模型直接回答，不使用 schools.json，不走图
    no_verify  完整系统，但跳过 verify，直接把草稿作为最终回答

指标（每次运行都计算，最后输出每次的值和平均值）：
    路由准确率      实际节点路径（去掉带冒号的内部条目和 START/END）与 expected_path 或 expected_path_any 之一一致；b0 不计
    事实准确率      A、C 类：每组 must_contain 至少出现一个词，且不触发 must_not_contain
    幻觉率          所有题目中触发 must_not_contain 的比例；F 类单独统计
    符合度判断正确率 B 类：per_school 中能自动判定的项与回答里该校的判断一致的比例；不能自动判定的写入结果文件供人工检查
    平均延迟、平均模型调用次数、异常率

结果写入 eval/results/<配置>_<时间>.json。需要真实模型时读取 ANTHROPIC_API_KEY（环境变量、.env 或 st.secrets）；
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
CONFIGS = ("full", "b0", "no_verify")

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
            elif actual is None and config == "b0":
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
    if config == "b0":
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


def invoke_graph(app: Any, question: str) -> tuple[dict, Optional[int]]:
    """运行图，返回 (最终状态, school_matcher 生成的补充说明条数；未经过 school_matcher 为 None)。"""
    final: dict = {}
    generated = None
    for mode, chunk in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        elif "school_matcher" in chunk:
            generated = (chunk["school_matcher"].get("draft") or "").count(g.SUPPLEMENT_LABEL)
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
        if config == "b0":
            response = counter.invoke([SystemMessage(content=B0_SYSTEM_PROMPT), HumanMessage(content=item["input"])])
            answer = text_of(response.content)
        else:
            result, generated = invoke_graph(app, item["input"])
            answer = final_answer(result)
            trace = result.get("trace", [])
            record["supplements"] = supplement_removal(trace, generated, result.get("draft", ""))
        record["error"] = None
    except Exception as e:  # 只记录异常类型，异常内容可能包含请求细节
        answer, record["error"] = "", type(e).__name__
        trace = [] if config != "b0" else None
    record["latency_s"] = round(time.perf_counter() - start, 3)
    record["model_calls"] = counter.calls if counter else 0
    record["answer"] = answer
    record["trace"] = trace
    record["path"] = node_path(trace) if trace is not None else None

    if config == "b0":
        hits = [p for p in DECLINE_PHRASES if p in answer]
        record["declined"] = {"ok": bool(hits), "hits": hits}

    prefix = item["id"][0]
    record["route"] = judge_route(item, record["path"]) if config != "b0" else {"ok": None, "reason": "b0 不计"}
    record["contains"] = judge_contains(answer, item["gold"])
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
        "b0_declined": sum(r["declined"]["ok"] for r in declined) if declined else None,
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
    ("b0_declined", "b0 表示无法回答的题数", "num"),
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="在 eval/testset.json 上评测")
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--repeat", type=int, default=1, help="重复运行次数")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 K 题（在跳过 needs_check 之后计数）")
    parser.add_argument("--include-unchecked", action="store_true", help="包含 needs_check 为 true 的题目")
    args = parser.parse_args(argv)

    print(api_key_hint())
    real_llm = get_llm()
    if real_llm is None and args.config == "b0":
        print("b0 配置需要模型，未检测到密钥，已退出。")
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
