"""live 冒烟测试：调用真实模型，检查各节点在真实模型下的行为，并统计每个用例的模型调用次数。

不在 pytest 默认运行范围内（文件名不以 test_ 开头），需要手动运行：
    .venv/bin/python smoke_live.py

需要 ANTHROPIC_API_KEY（环境变量、.env 或 st.secrets），每次运行会产生少量 API 费用。
只输出是否配置了密钥，不输出密钥本身。
"""

import copy
import sys

import graph as g
from config import api_key_hint
from llm import MODEL_NAME, get_llm, with_max_tokens

FORBIDDEN = ["不需要", "无需", "不要求", "不接受"]


class CountingLLM:
    """包装真实模型，统计 with_structured_output(...).invoke(...) 的调用次数。"""

    def __init__(self, llm):
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


# ---------------------------------------------------------------------------
# 检查函数
# ---------------------------------------------------------------------------

def verified(r):
    # 模型逐条判断通过（llm_pass）、关键陈述全是学校记录原文由代码判定（no_claims），
    # 或选校补充说明未通过已删除、回答其余部分照常输出（supplement_removed）
    return r["verdict"] == "pass" and any(
        t in ("verify:llm_pass", "verify:no_claims") or t.startswith("verify:supplement_removed(")
        for t in r["trace"]
    )


def no_admission_claims(r):
    return not g.admission_claims(r["draft"])


def check_in_library_deadline(r):
    return verified(r) and "2026-11-06" in r["draft"]


def check_out_of_library(r):
    # 库外学校：检索不到 → fallback，不应进入 verify，也就不调用模型核对
    return r["trace"][-1] == "fallback" and "verify" not in r["trace"]


def check_gmat_unmentioned(r):
    return verified(r) and "官网未说明" in r["draft"] and not any(p in r["draft"] for p in FORBIDDEN)


def check_gmat_optional(r):
    return verified(r) and "选择性提交" in r["draft"]


def routed(intent):
    def check(r):
        return r["intent"] == intent and "router:llm" in r["trace"]
    return check


def check_profile_complete(r):
    p = r["profile"]
    return (
        verified(r)
        and "profile_extractor:llm" in r["trace"]
        and not any(t.startswith("profile_extractor:dropped") for t in r["trace"])
        and "3.6" in p.get("gpa", "")
        and "7" in p.get("language_score", "")
        and "金融" in p.get("major", "")
        and "gmat_gre" not in p  # 没说过，不得推测
        and "school_matcher" in r["trace"]
        and no_admission_claims(r)
    )


def check_profile_missing(r):
    # 信息缺失时走 clarify → END，路径上没有 verify；检查不进入 fallback、不推测缺失字段
    p = r["profile"]
    return (
        "profile_extractor:llm" in r["trace"]
        and "3.4" in p.get("gpa", "")
        and "language_score" not in p and "major" not in p
        and r["trace"][-1] == "clarify"
        and "fallback" not in r["trace"]
    )


ESSAY = (
    "我本科就读于工商管理专业，大三在一家物流公司实习时，负责整理仓储周报，"
    "发现库存周转数据长期没有被用于决策。我用三周时间整理了过去一年的出入库记录，"
    "提出调整补货周期的建议，被部门采纳后试行了两个月。这段经历让我意识到，"
    "经营决策需要数据分析与管理视角的结合。我希望在经营学硕士阶段系统学习运营管理和数据分析，"
    "毕业后从事供应链管理相关工作，并希望将来回到中韩贸易相关的企业，帮助企业改进库存与采购流程。"
)


def check_review_llm(r):
    return (
        "doc_reviewer:llm" in r["trace"]
        and all(s in r["draft"] for s in ("【结构】", "【针对性】", "【具体性】"))
        and ESSAY not in r["draft"]  # 不改写/不复述全文
    )


def check_review_too_short(r):
    return "doc_reviewer:too_short" in r["trace"] and "doc_reviewer:llm" not in r["trace"]


def language_fit_line(r, school):
    """回答中某校的"语言：<符合度>"行（由代码判断）。"""
    para = next((p for p in r["draft"].split("\n\n") if p.startswith(f"{school} ")), "")
    return next((line.strip() for line in para.splitlines() if line.strip().startswith("语言：")), "")


def check_match_with(expected_fits):
    """选校回答：核对通过、调用了模型写补充说明、列出全部学校、不含录取概率措辞，
    且各校语言符合度与预期一致（expected_fits: {学校: 符合度前缀}）。"""
    def check(r):
        return (
            verified(r)
            and "school_matcher:llm" in r["trace"]
            and all(s["school"] in r["draft"] for s in g.load_schools())
            and no_admission_claims(r)
            and all(language_fit_line(r, school).startswith(f"语言：{fit}") for school, fit in expected_fits.items())
        )
    return check


UNSTATED = {"仁济大学": "官网未说明", "国民大学": "官网未说明"}
FITS_IELTS = {"庆熙大学": "无法判断（", "成均馆大学": "无法判断，以简章为准", "崇实大学": "无法判断，以简章为准", **UNSTATED}
FITS_TOPIK4 = {"庆熙大学": "不符合（", "崇实大学": "符合（", "成均馆大学": "无法判断，以简章为准", **UNSTATED}
FITS_TOEFL = FITS_IELTS
FITS_TOPIK6 = {"庆熙大学": "符合（", "崇实大学": "符合（", "成均馆大学": "无法判断，以简章为准", **UNSTATED}
FITS_TOPIK3 = {"庆熙大学": "不符合（", "崇实大学": "符合（", "成均馆大学": "无法判断，以简章为准", **UNSTATED}


def check_profile_complete_with_fits(r):
    return check_profile_complete(r) and check_match_with(FITS_IELTS)(r)


CASES = [
    ("库内学校的截止日期", "成均馆大学的申请截止日期是哪天？", check_in_library_deadline),
    ("库外学校", "首尔大学的申请要求是什么？", check_out_of_library),
    ("是否要 GMAT（官网未说明）", "国民大学需要 GMAT 吗？", check_gmat_unmentioned),
    ("是否要 GMAT（有原文）", "成均馆大学需要 GMAT 吗？", check_gmat_optional),
    ("router：闲聊", "早上好呀", routed("chitchat")),
    ("router：文书点评", "能帮我改一下个人陈述的开头吗？", routed("review")),
    ("router：无关问题", "下周首尔天气怎么样？", routed("unclear")),
    ("画像：信息完整", "我本科学金融，GPA 3.6/4.0，雅思 7.0，想申请韩国的经营学硕士，帮我选校", check_profile_complete_with_fits),
    ("画像：信息缺失", "我GPA 3.4，想申请经营大学院，帮我选校", check_profile_missing),
    ("文书：正常长度", f"帮我点评一下这篇个人陈述：{ESSAY}", check_review_llm),
    ("文书：太短", "帮我点评一下我的文书：我想读MBA，因为我喜欢管理。", check_review_too_short),
    ("选校：计算机背景（TOPIK 4级）", "我本科是计算机专业，GPA 3.2/4.0，TOPIK 4级，帮我选校", check_match_with(FITS_TOPIK4)),
    ("选校：会计背景（托福 95）", "我本科会计，GPA 3.7/4.0，托福 95，GMAT 650，帮我推荐学校", check_match_with(FITS_TOEFL)),
    ("选校：语言达标（TOPIK 6级）", "我本科经济学，GPA 3.5/4.0，TOPIK 6级，帮我选校", check_match_with(FITS_TOPIK6)),
    ("选校：语言不达标（TOPIK 3级）", "我本科经济学，GPA 3.5/4.0，TOPIK 3级，帮我选校", check_match_with(FITS_TOPIK3)),
]


def supplement_info(r) -> str:
    """最终回答中保留的补充说明数量，以及被删除或未通过代码检查的学校。"""
    total = len(r.get("docs") or [])
    shown = r["draft"].count("补充说明：")
    removed = [t[len("verify:supplement_removed("):-1] for t in r["trace"] if t.startswith("verify:supplement_removed(")]
    dropped = [t[len("school_matcher:reason_dropped("):-1] for t in r["trace"] if t.startswith("school_matcher:reason_dropped(")]
    parts = [f"显示 {shown}/{total} 所"]
    if dropped:
        parts.append(f"代码检查未通过：{'、'.join(dropped)}")
    if removed:
        parts.append(f"verify 未通过已删除：{'、'.join(removed)}")
    return "；".join(parts)


def main() -> int:
    print(api_key_hint())
    real = get_llm()
    if real is None:
        return 2
    print(f"模型：{MODEL_NAME}\n")

    failed = total_calls = 0
    for name, question, check in CASES:
        counter = CountingLLM(real)
        g.get_llm = lambda: counter
        result = g.run(question)
        ok = check(result)
        failed += not ok
        total_calls += counter.calls
        print(f"[{'PASS' if ok else 'FAIL'}] {name}（模型调用 {counter.calls} 次）")
        if "school_matcher" in result["trace"]:
            print(f"  补充说明: {supplement_info(result)}")
        print(f"  问题  : {question}")
        print(f"  trace : {' -> '.join(result['trace'])}")
        print(f"  回答  : {result['messages'][-1].content}\n")

    print(f"{len(CASES) - failed}/{len(CASES)} 通过，模型调用共 {total_calls} 次")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
