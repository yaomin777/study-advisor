"""AI经营学硕士留学申请顾问 —— LangGraph 总体架构（报告 2.1 节）。

9 个节点：router, profile_extractor, clarify, school_matcher, requirement_retriever,
doc_reviewer, timeline_planner, verify, fallback。
4 条条件边：route_by_intent, route_after_profile, route_after_retrieval, route_after_verify。

节点以规则版为主；verify 在配置了密钥时额外调用模型核对（见 llm.py），无密钥时只做代码核对。
"""

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from llm import VERIFY_MAX_TOKENS, get_llm, with_max_tokens

MAX_RETRIES = 2

# 申请者画像中必须具备的字段（GMAT/GRE 有则记录，不追问）
REQUIRED_PROFILE_FIELDS = ["gpa", "language_score", "major"]

# 学校数据（格式见 data/schools.json 中的 "_format"）。后续由 RAG 检索替代。
# 注意：只收录有官方出处的数据，不编造学校要求或截止日期。
SCHOOLS_PATH = Path(__file__).parent / "data" / "schools.json"


def load_schools(path: Optional[Path] = None) -> list[dict[str, Any]]:
    path = path or SCHOOLS_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("schools", [])


def schools_from(config: Optional[RunnableConfig]) -> list[dict[str, Any]]:
    """优先使用调用方传入的学校数据（如 Streamlit 缓存），否则读取文件。"""
    schools = ((config or {}).get("configurable") or {}).get("schools")
    return load_schools() if schools is None else schools


def matches_school(record: dict[str, Any], question: str) -> bool:
    """问题中是否提到该校（school 或 aliases）。英文/数字别名按整词匹配，避免 SSU 命中 issue。"""
    for name in [record.get("school", "")] + record.get("aliases", []):
        if not name:
            continue
        if name.isascii():
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", question, re.IGNORECASE):
                return True
        elif name in question:
            return True
    return False


def get_today() -> date:
    return date.today()


def round_status(deadline: str, today: date) -> str:
    d = date.fromisoformat(deadline)
    if d < today:
        return "已截止（以最新简章为准）"
    if d == today:
        return "今日截止"
    return f"未截止（距截止 {(d - today).days} 天）"


def upcoming_rounds(record: dict[str, Any], today: date) -> list[dict[str, str]]:
    return [r for r in record.get("rounds", []) if date.fromisoformat(r["deadline"]) >= today]


def format_rounds(record: dict[str, Any], today: date) -> str:
    rounds = record.get("rounds", [])
    return (
        "；".join(f"{r['name']} 截止 {r['deadline']}，{round_status(r['deadline'], today)}" for r in rounds)
        or "资料中未提供轮次日期"
    )


GMAT_GRE_RE = re.compile(r"(?<![A-Za-z0-9])(GMAT|GRE)(?![A-Za-z0-9])", re.IGNORECASE)
GMAT_GRE_UNMENTIONED = "资料中未提及，请以最新简章为准"


def format_gmat_gre(record: dict[str, Any]) -> str:
    """资料为"未提及"时如实说明，不推断为"不需要"。"""
    text = record.get("gmat_gre", "")
    if not text or "未提及" in text:
        return GMAT_GRE_UNMENTIONED
    return f"{text}（请以最新简章为准）"


WORK_EXPERIENCE_RE = re.compile(r"工作经验|工作经历|工作年限|work experience", re.IGNORECASE)


def format_work_experience(record: dict[str, Any]) -> str:
    """按数据原文回答，不改写成"不接受"或"不需要"。"""
    return record.get("work_experience") or GMAT_GRE_UNMENTIONED


def format_school(record: dict[str, Any], today: date) -> str:
    return (
        f"{record['school']} {record['program']}\n"
        f"  申请轮次（今天 {today.isoformat()}）：{format_rounds(record, today)}\n"
        f"  GMAT/GRE：{format_gmat_gre(record)}\n"
        f"  语言：{record.get('language', '')}\n"
        f"  背景要求：{record.get('background_req', '')}\n"
        f"  文书：{'；'.join(record.get('essays', [])) or '未提及'}\n"
        f"  推荐信：{record.get('recommendations', '')}\n"
        f"  工作经验：{format_work_experience(record)}\n"
        f"  出处：{record['source_url']}（截至 {record.get('as_of', '')}）"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AdvisorState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史
    intent: str                     # profile / qa / timeline / review / unclear / chitchat
    question: str                   # 本轮用户问题
    profile: dict[str, Any]         # 申请者画像
    missing_fields: list[str]       # 画像中缺失的字段
    matched_schools: dict[str, dict[str, str]]  # {学校: {"语言": 符合度, "本科专业": ..., ...}}
    docs: list[dict[str, str]]      # 检索到的文档
    sources: list[str]              # 出处（URL 等）
    draft: str                      # 当前生成的回答草稿
    verdict: str                    # verify 结果：pass / fail
    last_agent: str                 # 最近一个生成回答的节点，用于 verify 失败后重试
    retry_count: int                # 已重试次数
    trace: list[str]                # 经过的节点名称


def _trace(state: AdvisorState, node: str) -> list[str]:
    return state.get("trace", []) + [node]


def _retry_count(state: AdvisorState) -> int:
    """生成节点在 verify 判定失败后被重新调用时，重试次数 +1。"""
    count = state.get("retry_count", 0)
    return count + 1 if state.get("verdict") == "fail" else count


# ---------------------------------------------------------------------------
# Nodes（占位实现）
# ---------------------------------------------------------------------------

INTENT_KEYWORDS = [
    ("review", ["文书", "个人陈述", "PS", "SOP", "essay", "润色", "点评", "修改简历"]),
    ("timeline", ["时间规划", "时间线", "规划", "timeline", "什么时候", "几月"]),
    ("profile", ["选校", "推荐学校", "定位", "能申请", "背景", "GPA"]),
    ("qa", ["申请要求", "要求", "雅思", "托福", "IELTS", "TOEFL", "GMAT", "GRE", "材料", "截止", "deadline",
            "工作经验", "工作经历"]),
    ("chitchat", ["你好", "您好", "谢谢", "hello", "hi", "thanks"]),
]


INTENTS = ("profile", "qa", "timeline", "review", "chitchat", "unclear")


class RouterResult(BaseModel):
    intent: Literal["profile", "qa", "timeline", "review", "chitchat", "unclear"] = Field(
        description="用户意图"
    )


ROUTER_SYSTEM_PROMPT = """你是经营学硕士留学申请顾问系统的意图分类器。根据用户的最新一句话，从下列意图中选择一个：

- profile：提供个人背景（GPA、语言成绩、GMAT/GRE、本科专业等），或请求选校、定位、推荐学校
- qa：询问某所学校的具体申请信息，如申请要求、材料、语言或 GMAT/GRE 要求、工作经验、截止日期
- timeline：请求制定申请时间规划或时间线
- review：请求点评、修改或润色文书、个人陈述、简历
- chitchat：打招呼、致谢等闲聊
- unclear：意图不明确，或与留学申请无关

只输出意图，不回答问题。"""


def keyword_intent(question: str) -> str:
    lowered = question.lower()
    for name, keywords in INTENT_KEYWORDS:
        if any(k.lower() in lowered for k in keywords):
            return name
    return "unclear"


def llm_intent(llm: Any, question: str) -> tuple[Optional[str], str]:
    """返回 (intent, trace 条目)；intent 为 None 表示需退回关键词版。"""
    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=question)]
    try:
        result = llm.with_structured_output(RouterResult, method="json_schema").invoke(messages)
    except OutputParserException:
        return None, "router:llm_parse_error"
    except Exception:  # 超时、网络等；不记录异常内容
        return None, "router:llm_error"
    if not isinstance(result, RouterResult):
        return None, "router:llm_parse_error"
    if result.intent not in INTENTS:
        return None, "router:llm_invalid_intent"
    return result.intent, "router:llm"


def router(state: AdvisorState) -> dict:
    """识别用户意图：有密钥时由模型分类，模型出错或结果不在范围内时退回关键词版。"""
    question = state.get("question") or ""
    if not question:
        human = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        question = human[-1].content if human else ""

    trace = ["router"]
    intent = None
    llm = get_llm()
    if llm is not None:
        intent, entry = llm_intent(llm, question)
        trace.append(entry)
    if intent is None:
        intent = keyword_intent(question)

    update = {
        "intent": intent,
        "question": question,
        # 每轮重置与本轮回答相关的字段
        "docs": [],
        "sources": [],
        "draft": "",
        "verdict": "",
        "last_agent": "",
        "retry_count": 0,
        "trace": trace,
    }
    if intent == "chitchat":
        reply = "你好！我是 AI 经营学硕士留学申请顾问，可以帮你做选校定位、查询申请要求、规划时间线和点评文书。"
        update["draft"] = reply
        update["messages"] = [AIMessage(content=reply)]
    return update


REGION_KEYWORDS = ["美国", "英国", "香港", "新加坡", "澳洲", "澳大利亚", "加拿大", "韩国", "日本", "欧洲"]


PROFILE_FIELDS = ("gpa", "language_score", "gmat_gre", "major")


class ProfileResult(BaseModel):
    gpa: Optional[str] = Field(description='GPA，写成 "3.6"，用户说了满分制时写成 "3.6/4.0"；用户没说为 null')
    language_score: Optional[str] = Field(
        description='语言成绩，写成 "考试名 分数或级别"，如 "IELTS 7.0"、"TOEFL 95"、"TOPIK 4"；用户没说为 null'
    )
    gmat_gre: Optional[str] = Field(description='写成 "GMAT 700" 或 "GRE 320"；用户没说为 null')
    major: Optional[str] = Field(description="本科专业，照抄用户原话中的专业名称；用户没说为 null")


PROFILE_SYSTEM_PROMPT = """你是经营学硕士留学申请顾问系统的信息抽取器。从用户这句话中抽取申请者画像：GPA、语言成绩、GMAT/GRE、本科专业。

规则：
1. 只抽取用户明确说出的内容。用户没说的字段一律为 null，不得推测、换算或补全。
2. 不要根据学校、专业或其他信息推断分数；不要把目标专业当作本科专业。
3. 数字照抄用户说的数值；本科专业照抄用户原话中的专业名称。"""


def rule_profile_fields(text: str) -> dict[str, Optional[str]]:
    """规则版抽取（正则），模型不可用时使用。"""
    fields: dict[str, Optional[str]] = dict.fromkeys(PROFILE_FIELDS)
    if m := re.search(r"GPA\s*[:：是为]?\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?", text, re.IGNORECASE):
        fields["gpa"] = f"{m.group(1)}/{m.group(2)}" if m.group(2) else m.group(1)
    if m := re.search(r"(?:本科(?:专业)?|专业)\s*[:：是为]?\s*([一-龥A-Za-z]+?)(?:专业|，|,|。|\s|$)", text):
        fields["major"] = m.group(1)
    if m := re.search(r"(雅思|IELTS|托福|TOEFL)\s*[:：是为]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE):
        fields["language_score"] = f"{m.group(1)} {m.group(2)}"
    elif m := re.search(r"TOPIK\s*[:：是为]?\s*(\d)\s*级?", text, re.IGNORECASE):
        fields["language_score"] = f"TOPIK {m.group(1)}"
    if m := re.search(r"(GMAT|GRE)\s*[:：是为]?\s*(\d+)", text, re.IGNORECASE):
        fields["gmat_gre"] = f"{m.group(1).upper()} {m.group(2)}"
    return fields


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
EXAM_SYNONYMS = {"IELTS": ["ielts", "雅思"], "TOEFL": ["toefl", "托福"], "TOPIK": ["topik"], "GMAT": ["gmat"], "GRE": ["gre"]}


def _mentions(term: str, text: str) -> bool:
    if term.isascii():
        return re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text, re.IGNORECASE) is not None
    return term in text


def _supported_by_text(field: str, value: str, text: str) -> bool:
    """模型抽取的值必须能在用户原话中找到：数字须出现，考试名须提到，专业须照抄。"""
    if field == "major":
        return value.lower() in text.lower()
    numbers = NUMBER_RE.findall(value)
    text_numbers = {float(n) for n in NUMBER_RE.findall(text)}
    if not numbers or any(float(n) not in text_numbers for n in numbers):
        return False
    if field in ("language_score", "gmat_gre"):
        parsed = parse_language_score(value) if field == "language_score" else None
        exam = parsed[0] if parsed else re.split(r"[\s\d]", value.strip())[0]
        canonical = LANGUAGE_ALIASES.get(exam, exam.upper())
        terms = EXAM_SYNONYMS.get(canonical, [exam])
        return bool(exam) and any(_mentions(t, text) for t in terms)
    return True


def llm_profile_fields(llm: Any, text: str) -> tuple[Optional[dict[str, Optional[str]]], list[str]]:
    """返回 (字段, trace 条目)；字段为 None 表示需退回规则版。原话中找不到依据的值置为 null。"""
    messages = [SystemMessage(content=PROFILE_SYSTEM_PROMPT), HumanMessage(content=text)]
    try:
        result = llm.with_structured_output(ProfileResult, method="json_schema").invoke(messages)
    except OutputParserException:
        return None, ["profile_extractor:llm_parse_error"]
    except Exception:  # 超时、网络等；不记录异常内容
        return None, ["profile_extractor:llm_error"]
    if not isinstance(result, ProfileResult):
        return None, ["profile_extractor:llm_parse_error"]

    fields: dict[str, Optional[str]] = {}
    entries = ["profile_extractor:llm"]
    for f in PROFILE_FIELDS:
        value = (getattr(result, f) or "").strip() or None
        if value and not _supported_by_text(f, value, text):
            entries.append(f"profile_extractor:dropped({f})")
            value = None
        fields[f] = value
    return fields, entries


def profile_extractor(state: AdvisorState) -> dict:
    """提取申请者画像：有密钥时由模型抽取，模型出错时退回正则版。只记录用户明确说过的内容。"""
    text = state.get("question", "")
    profile = dict(state.get("profile") or {})
    trace = _trace(state, "profile_extractor")

    fields = None
    llm = get_llm()
    if llm is not None:
        fields, entries = llm_profile_fields(llm, text)
        trace += entries
    if fields is None:
        fields = rule_profile_fields(text)
    # 本轮没说的字段保留之前的值
    profile.update({f: v for f, v in fields.items() if v})

    # GMAT/GRE、目标地区为可选信息，有则记录，不计入 missing_fields
    regions = [r for r in REGION_KEYWORDS if r in text]
    if regions:
        profile["target_region"] = "、".join(regions)

    missing = [f for f in REQUIRED_PROFILE_FIELDS if not profile.get(f)]
    return {
        "profile": profile,
        "missing_fields": missing,
        "trace": trace,
    }


FIELD_LABELS = {
    "gpa": "GPA（及满分制）",
    "language_score": "语言成绩（雅思/托福/TOPIK）",
    "major": "本科专业",
}


def clarify(state: AdvisorState) -> dict:
    """向用户追问缺失信息。"""
    missing = state.get("missing_fields") or []
    if state.get("intent") == "unclear" or not missing:
        reply = "能再具体说说你的需求吗？例如：选校定位、查询某校申请要求、制定申请时间规划，或点评文书。"
    else:
        items = "、".join(FIELD_LABELS.get(f, f) for f in missing)
        reply = f"为了给出选校建议，还需要你补充以下信息：{items}。"
    return {
        "draft": reply,
        "last_agent": "clarify",
        "messages": [AIMessage(content=reply)],
        "trace": _trace(state, "clarify"),
    }


LANGUAGE_ALIASES = {"雅思": "IELTS", "托福": "TOEFL"}
LANGUAGE_SCORE_RE = re.compile(r"\s*([^\d\s]*)\s*(\d+(?:\.\d+)?)\s*(级|分)?\s*")


def parse_language_score(score: str) -> Optional[tuple[str, float]]:
    """解析为 (考试名, 分数)。支持 "IELTS 7.0" "雅思7分" "TOPIK 4级" "TOPIK4" "4级" "4 级"；
    只写"N级"时按 TOPIK 处理。"""
    m = LANGUAGE_SCORE_RE.fullmatch(score or "")
    if not m:
        return None
    name, value, unit = m.group(1), float(m.group(2)), m.group(3)
    if not name:
        return ("TOPIK", value) if unit == "级" else None
    return LANGUAGE_ALIASES.get(name, name.upper()), value


# ---------------------------------------------------------------------------
# 逐校要求符合度：由代码根据学校记录判断，结果只能是 符合 / 不符合 / 无法判断 / 官网未说明
# ---------------------------------------------------------------------------

FIT_YES, FIT_NO, FIT_UNKNOWN, FIT_UNSTATED = "符合", "不符合", "无法判断", "官网未说明"
FIT_BY_BROCHURE = f"{FIT_UNKNOWN}，以简章为准"
MAJOR_UNRESTRICTED = "未限制专业"
UNSTATED_MARKERS = ("官网未说明", "官网未提及", "未提及")
MAJOR_UNRESTRICTED_MARKERS = ("未限制专业", "未限制本科专业", "与所修专业无关")
ENGLISH_TEST_WORDS = ("TOEFL", "TOEIC", "IELTS", "TEPS", "OPIc", "英语")
TOPIK_REQUIREMENT_RE = re.compile(r"TOPIK\s*(\d)\s*级以上")


def topik_minimum(text: str) -> Optional[int]:
    """记录中报考所需的 TOPIK 最低级别；"毕业需 TOPIK N级以上"这类毕业要求不计。"""
    for m in TOPIK_REQUIREMENT_RE.finditer(text):
        if "毕业" not in text[max(0, m.start() - 6):m.start()]:
            return int(m.group(1))
    return None


def language_fit(record: dict[str, Any], profile: dict[str, Any]) -> str:
    text = record.get("language") or ""
    if not text or any(k in text for k in UNSTATED_MARKERS):
        return FIT_UNSTATED
    parsed = parse_language_score(profile.get("language_score") or "")
    if parsed is None:
        return f"{FIT_UNKNOWN}（未提供可识别的语言成绩）"
    exam, value = parsed
    minimum = topik_minimum(text)
    if exam == "TOPIK":
        if minimum is None:
            return f"{FIT_BY_BROCHURE}（记录未列出 TOPIK 级别要求）"
        verdict = FIT_YES if value >= minimum else FIT_NO
        return f"{verdict}（要求 TOPIK {minimum}级以上，你的 TOPIK {value:g}级）"
    if any(w.lower() in text.lower() for w in ENGLISH_TEST_WORDS):
        return f"{FIT_BY_BROCHURE}（记录未给出 {exam} 的具体分数要求）"
    if minimum is not None:
        return f"{FIT_UNKNOWN}（要求 TOPIK {minimum}级以上，未提供 TOPIK 成绩）"
    return FIT_BY_BROCHURE


def major_fit(record: dict[str, Any]) -> str:
    background = record.get("background_req") or ""
    return MAJOR_UNRESTRICTED if any(k in background for k in MAJOR_UNRESTRICTED_MARKERS) else FIT_UNSTATED


def school_fit(record: dict[str, Any], profile: dict[str, Any], today: date) -> dict[str, str]:
    return {
        "语言": language_fit(record, profile),
        "本科专业": major_fit(record),
        "GMAT/GRE": record.get("gmat_gre") or FIT_UNSTATED,
        "工作经验": record.get("work_experience") or FIT_UNSTATED,
        "截止状态": format_rounds(record, today),
    }


# ---------------------------------------------------------------------------
# 模型补充说明（可选）：只作为每所学校的补充，仍由 verify 核对，未通过就不显示
# ---------------------------------------------------------------------------

class SchoolReason(BaseModel):
    school: str = Field(description="学校名称，与输入完全一致")
    reason: str = Field(description="补充说明，不超过 60 字")


class MatchReasons(BaseModel):
    reasons: list[SchoolReason] = Field(description="每所学校一条补充说明")


MATCH_SYSTEM_PROMPT = """你是经营学硕士选校顾问。系统已用代码判断每所学校各项公开要求的符合度，你只为每所学校写一句补充说明。

规则：
1. 只能使用该校"资料"中已有的字段（背景要求、语言、GMAT/GRE、文书、推荐信、工作经验等）和"申请者画像"中已有的字段。
   资料里没有的信息一律不写，例如 GPA 门槛、GMAT/GRE 对评审的影响、录取率、排名、分数线、学费。
2. 记录里写"具体分数以简章为准"或没有给出具体分数时，不得判断用户的英语成绩是否达标。
3. 不得与"系统判断的符合度"矛盾。不得断言录取概率或录取把握（如"很可能录取""录取把握大"），
   只能说明用户符合或不符合哪项公开要求；比较时写明双方数据，例如"语言要求 TOPIK 5级以上，你的 TOPIK 4级未达到"。
4. 资料写"官网未说明"或"未提及"时，只能如实说明，不能说成"不要求""不需要"。
5. 不要写截止状态（已截止、尚未截止、距截止几天），这部分由系统给出。
6. 不要陈述用户没有某项成绩或某项条件（例如"你没有 GMAT 成绩"）。
7. 学校记录里没有具体分数标准时，不要判断用户的成绩是否达标，也不要判断某个分数算不算优秀。
8. 每所学校一条，学校名称与输入完全一致，每条不超过 60 字，用中文。"""

MAX_REASON_CHARS = 120


def reason_supported(reason: str, record: dict[str, Any], profile: dict[str, Any]) -> bool:
    """补充说明中的数字、日期和否定说法必须能在该校记录或申请者画像中找到，且不得断言录取概率。"""
    if not reason.strip() or len(reason) > MAX_REASON_CHARS:
        return False
    if unsupported_dates(reason, [record], get_today()) or unsupported_negations(reason, [record]):
        return False
    if admission_claims(reason):
        return False
    source_text = json.dumps(record, ensure_ascii=False) + json.dumps(profile, ensure_ascii=False)
    allowed = {float(n) for n in NUMBER_RE.findall(source_text)}
    return all(float(n) in allowed for n in NUMBER_RE.findall(reason))


def llm_match_reasons(
    llm: Any, profile: dict[str, Any], schools: list[dict[str, Any]], fits: dict[str, dict[str, str]]
) -> tuple[dict[str, str], list[str]]:
    """返回 {学校: 补充说明}（只含通过代码检查的）与 trace 条目；出错时返回空字典。"""
    payload = {
        "申请者画像": {f: profile.get(f) for f in PROFILE_FIELDS},
        "学校": [
            {"资料": {k: v for k, v in r.items() if k != "aliases"}, "系统判断的符合度": fits[r["school"]]}
            for r in schools
        ],
    }
    messages = [
        SystemMessage(content=MATCH_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=1)),
    ]
    try:
        result = llm.with_structured_output(MatchReasons, method="json_schema").invoke(messages)
    except OutputParserException:
        return {}, ["school_matcher:llm_parse_error"]
    except Exception:  # 超时、网络等；不记录异常内容
        return {}, ["school_matcher:llm_error"]
    if not isinstance(result, MatchReasons):
        return {}, ["school_matcher:llm_parse_error"]

    given = {r.school: r.reason.strip() for r in result.reasons}
    reasons, entries = {}, ["school_matcher:llm"]
    for record in schools:
        name = record["school"]
        if name in given and reason_supported(given[name], record, profile):
            reasons[name] = given[name]
        else:
            entries.append(f"school_matcher:reason_dropped({name})")
    return reasons, entries


SUPPLEMENT_LABEL = "补充说明："
MATCH_HEADER = "以下为各校公开要求的符合度（由系统根据学校记录判断，仅供参考，请以最新简章为准）："


def school_header(record: dict[str, Any]) -> str:
    return f"{record.get('school', '')} {record.get('program', '')}".strip()


def format_match_answer(
    profile: dict[str, Any], schools: list[dict[str, Any]], fits: dict[str, dict[str, str]], supplements: dict[str, str]
) -> str:
    lines = [
        f"你的画像：GPA {profile.get('gpa') or '未提供'}（学校记录未提供 GPA 要求，未作判断）；"
        f"语言成绩 {profile.get('language_score') or '未提供'}；本科专业 {profile.get('major') or '未提供'}；"
        f"GMAT/GRE {profile.get('gmat_gre') or '未提供'}",
        MATCH_HEADER,
    ]
    for r in schools:
        fit = fits[r["school"]]
        lines += [
            "",
            school_header(r),
            f"  语言：{fit['语言']}｜原文：{r.get('language') or FIT_UNSTATED}",
            f"  本科专业：{fit['本科专业']}｜原文：{r.get('background_req') or FIT_UNSTATED}",
            f"  GMAT/GRE：{fit['GMAT/GRE']}",
            f"  工作经验：{fit['工作经验']}",
            f"  截止状态：{fit['截止状态']}",
        ]
        if r["school"] in supplements:
            lines.append(f"  {SUPPLEMENT_LABEL}{supplements[r['school']]}")
        lines.append(f"  出处：{r.get('source_url', '')}（数据整理日期 {r.get('as_of', '未知')}）")
    return "\n".join(lines)


def remove_supplements(draft: str, schools: list[dict[str, Any]], names: Optional[set[str]] = None) -> str:
    """删除指定学校（names 为 None 时为全部学校）的补充说明行，其余内容不变。"""
    headers = {school_header(d): d.get("school", "") for d in schools}
    kept, current = [], ""
    for line in draft.splitlines():
        if line in headers:
            current = headers[line]
        if line.strip().startswith(SUPPLEMENT_LABEL) and (names is None or current in names):
            continue
        kept.append(line)
    return "\n".join(kept)


def school_matcher(state: AdvisorState, config: Optional[RunnableConfig] = None) -> dict:
    """逐校列出公开要求的符合度（由代码根据学校记录判断：语言、本科专业、GMAT/GRE、工作经验、截止状态），
    引用原文并附出处；有密钥时由模型为每所学校写一句补充说明，补充说明由 verify 核对，未通过就不显示。"""
    profile = state.get("profile", {})
    schools = schools_from(config)
    today = get_today()
    fits = {s["school"]: school_fit(s, profile, today) for s in schools}
    trace = _trace(state, "school_matcher")

    supplements: dict[str, str] = {}
    llm = get_llm() if schools else None
    if llm is not None:
        supplements, entries = llm_match_reasons(llm, profile, schools, fits)
        trace += entries

    return {
        "matched_schools": fits,
        "docs": schools,
        "sources": [d["source_url"] for d in schools if d.get("source_url")],
        "draft": format_match_answer(profile, schools, fits, supplements),
        "last_agent": "school_matcher",
        "retry_count": _retry_count(state),
        "trace": trace,
    }


def requirement_retriever(state: AdvisorState, config: Optional[RunnableConfig] = None) -> dict:
    """检索学校申请要求（占位：在 data/schools.json 中按校名匹配，后续替换为 RAG）。"""
    question = state.get("question", "")
    docs = [d for d in schools_from(config) if matches_school(d, question)]
    sources = [d["source_url"] for d in docs if d.get("source_url")]

    update = {
        "docs": docs,
        "sources": sources,
        "last_agent": "requirement_retriever",
        "retry_count": _retry_count(state),
        "trace": _trace(state, "requirement_retriever"),
    }
    if state.get("intent") == "qa" and docs:
        # 问到具体项目时只回答该项
        topics = []
        if GMAT_GRE_RE.search(question):
            topics.append(("GMAT/GRE", format_gmat_gre))
        if WORK_EXPERIENCE_RE.search(question):
            topics.append(("工作经验", format_work_experience))
        if topics:
            body = "\n".join(
                f"- {d['school']} {d['program']}："
                + "；".join(f"{label} {fmt(d)}" for label, fmt in topics)
                + f"（出处：{d['source_url']}，截至 {d.get('as_of', '')}）"
                for d in docs
            )
        else:
            today = get_today()
            body = "\n".join(format_school(d, today) for d in docs)
        update["draft"] = f"根据检索到的资料：\n{body}"
    return update


MIN_ESSAY_CHARS = 150


class ReviewResult(BaseModel):
    structure: str = Field(description="结构：段落组织、逻辑顺序、开头与结尾")
    relevance: str = Field(description="针对性：是否回应目标项目，动机与职业目标是否与经营学硕士相关")
    specificity: str = Field(description="具体性：是否有具体经历、数据和例子支撑")


REVIEW_SYSTEM_PROMPT = """你是经营学硕士申请文书顾问。阅读用户提供的文书，从结构、针对性、具体性三方面各给出反馈。

规则：
1. 每方面先指出一个做得好的地方，再给出一到两条可操作的改进建议，每方面不超过 120 字。
2. 不改写全文，不提供替换段落或范文；需要举例时最多引用原文中的一句话。
3. 只评价文书本身，不评价录取机会，不编造学校要求。
4. 用中文回答。"""

RULE_REVIEW = (
    "当前为规则版，无法逐句点评。请从以下三方面自查：\n"
    "【结构】开头是否点明申请动机；段落是否按“经历→收获→目标”展开；结尾是否回扣申请项目。\n"
    "【针对性】是否写明为什么选择该项目，以及项目与职业目标的联系。\n"
    "【具体性】经历是否有时间、角色、数据或结果支撑，避免只写形容词。"
)


def essay_text(question: str) -> str:
    """去掉"帮我看看这篇文书："之类的前缀，只保留文书正文。"""
    m = re.match(r"^[^：:\n]{0,40}[：:]\s*", question)
    if m and question[m.end():].strip():
        return question[m.end():]
    return question


def llm_review(llm: Any, essay: str) -> tuple[Optional[str], str]:
    """返回 (反馈, trace 条目)；反馈为 None 表示需退回规则版。"""
    messages = [SystemMessage(content=REVIEW_SYSTEM_PROMPT), HumanMessage(content=essay)]
    try:
        result = llm.with_structured_output(ReviewResult, method="json_schema").invoke(messages)
    except OutputParserException:
        return None, "doc_reviewer:llm_parse_error"
    except Exception:  # 超时、网络等；不记录异常内容
        return None, "doc_reviewer:llm_error"
    if not isinstance(result, ReviewResult) or not all(
        x.strip() for x in (result.structure, result.relevance, result.specificity)
    ):
        return None, "doc_reviewer:llm_parse_error"
    feedback = (
        "文书反馈（仅供参考，未改写原文）：\n"
        f"【结构】{result.structure.strip()}\n"
        f"【针对性】{result.relevance.strip()}\n"
        f"【具体性】{result.specificity.strip()}"
    )
    return feedback, "doc_reviewer:llm"


def doc_reviewer(state: AdvisorState) -> dict:
    """点评申请文书：从结构、针对性、具体性三方面给反馈，不改写全文。
    文本太短时提示补充；有密钥时由模型点评，出错时退回规则版自查清单。"""
    essay = essay_text(state.get("question", ""))
    length = len(re.sub(r"\s", "", essay))
    trace = _trace(state, "doc_reviewer")

    if length < MIN_ESSAY_CHARS:
        reply = (
            f"文书内容较短（约 {length} 字），无法给出有效反馈。"
            f"请粘贴完整的段落或全文（建议至少 {MIN_ESSAY_CHARS} 字）后再发给我。"
        )
        trace.append("doc_reviewer:too_short")
    else:
        reply = None
        llm = get_llm()
        if llm is not None:
            reply, entry = llm_review(llm, essay)
            trace.append(entry)
        if reply is None:
            reply = f"已收到文书内容（约 {length} 字）。{RULE_REVIEW}"

    return {
        "draft": reply,
        "last_agent": "doc_reviewer",
        "messages": [AIMessage(content=reply)],
        "trace": trace,
    }


def timeline_planner(state: AdvisorState) -> dict:
    """制定申请时间规划（占位：按当天日期标注轮次状态，不生成新日期）。"""
    docs = state.get("docs", [])
    today = get_today()
    lines = []
    for d in docs:
        line = (
            f"- {d['school']} {d['program']}：{format_rounds(d, today)}"
            f"（出处：{d['source_url']}，截至 {d.get('as_of', '')}）"
        )
        if d.get("rounds") and not upcoming_rounds(d, today):
            line += "\n  注意：资料中的轮次均已截止，暂无可申请轮次，请查阅最新简章确认下一轮日期。"
        lines.append(line)
    draft = (
        f"[占位] 今天是 {today.isoformat()}。以下为检索到的申请轮次信息，"
        f"完整的时间规划功能待接入大模型：\n" + "\n".join(lines)
    )
    return {
        "draft": draft,
        "last_agent": "timeline_planner",
        "retry_count": _retry_count(state),
        "trace": _trace(state, "timeline_planner"),
    }


SOURCE_REQUIRED_AGENTS = {"requirement_retriever", "timeline_planner"}


FULL_DATE_RE = re.compile(r"(\d{4})\s*(?:[-./]|年)\s*(\d{1,2})\s*(?:[-./]|月)\s*(\d{1,2})\s*日?")
MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日")

# 否定类说法：草稿中出现时，资料原文里必须有同样的说法，
# 防止把"官网未说明""未提及"写成"不要求""不需要"。
NEGATION_PHRASES = ("不需要", "不要求", "无需", "不用", "不必", "不接受")

# 断言录取概率的说法：选校理由只能说明符合或不符合哪项公开要求
ADMISSION_PHRASES = (
    "很可能录取", "大概率录取", "有望录取", "容易录取", "一定能录取", "保证录取", "稳录", "必录",
    "录取把握", "录取概率", "录取几率", "录取机会", "录取可能性",
)


def admission_claims(text: str) -> list[str]:
    return [p for p in ADMISSION_PHRASES if p in text]


def find_dates(text: str) -> tuple[set[str], set[tuple[int, int]]]:
    """返回完整日期（ISO 格式；非法日期保留原文）和缺少年份的"M月D日"。"""
    full: set[str] = set()

    def collect(m: re.Match) -> str:
        try:
            full.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat())
        except ValueError:
            full.add(m.group(0))
        return " "

    rest = FULL_DATE_RE.sub(collect, text)
    month_days = {(int(m), int(d)) for m, d in MONTH_DAY_RE.findall(rest)}
    return full, month_days


def unsupported_dates(draft: str, docs: list[dict[str, Any]], today: date) -> list[str]:
    """草稿中不在本次检索记录里的日期（当天日期除外）。"""
    allowed, _ = find_dates(json.dumps(docs, ensure_ascii=False))
    allowed.add(today.isoformat())
    allowed_month_days = {(int(d[5:7]), int(d[8:10])) for d in allowed if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)}

    full, month_days = find_dates(draft)
    bad = sorted(d for d in full if d not in allowed)
    bad += [f"{m}月{d}日" for m, d in sorted(month_days) if (m, d) not in allowed_month_days]
    return bad


def unsupported_negations(draft: str, docs: list[dict[str, Any]]) -> list[str]:
    docs_text = json.dumps(docs, ensure_ascii=False)
    return [p for p in NEGATION_PHRASES if p in draft and p not in docs_text]


class ClaimCheck(BaseModel):
    index: int = Field(description="待核对陈述的编号")
    evidence: list[str] = Field(
        description="从【学校记录】或【用户画像】中逐字摘录的依据原文；比较类陈述需同时摘录两边；找不到依据时为空列表"
    )
    reason: str = Field(description="一句话说明，不超过 30 字")


class VerifyResult(BaseModel):
    checks: list[ClaimCheck] = Field(description="对每条待核对陈述的判断，每条一次")


VERIFY_SYSTEM_PROMPT = """你是留学申请顾问系统的事实核对员，只根据用户消息中提供的资料工作，不使用任何外部知识。
资料分两类：【学校记录】和【用户画像】。【待核对陈述】是编号列表，每条前面的学校名称表示该陈述针对哪所学校。

对每条陈述输出一次：index（编号）、evidence（依据原文）、reason（一句话说明）。

evidence 的要求：
1. 必须从【学校记录】或【用户画像】中逐字摘录，不得改写、概括或拼接；可以摘录多段。
2. 学校要求类陈述（语言、GMAT/GRE、文书、推荐信、工作经验、背景或专业）：摘录该校记录中的对应原文。
3. 关于用户的内容：摘录【用户画像】中的对应值。
4. 用户与学校要求的比较：同时摘录学校记录一侧和用户画像一侧的原文。
5. 资料中找不到依据时，evidence 为空列表；以下情况也视为找不到依据，evidence 为空列表：
   记录写"官网未说明""官网未提及"而陈述说成"不要求""不需要""无需""不接受"；陈述断言录取概率或录取把握；
   比较方向错误（例如要求"5级以上"而陈述说 3级已达到）。

只输出列表中的编号，每条一次；不要输出列表以外的内容。"""


LLM_ERROR = "llm_error"
MAX_CLAIM_CHARS = 80
MAX_REASON_TRACE_CHARS = 40

# ---- 从草稿中取出待核对的关键陈述 ----
STATEMENT_SKIP_PREFIXES = ("根据检索到的资料", "出处", "申请轮次", "[占位]", "注意：", "文书反馈")
ANNOTATION_RE = re.compile(r"（请以最新简章为准）|（出处：[^）]*）|（截至[^）]*）")
TOPIC_LABEL_RE = re.compile(r"^(?:GMAT/GRE|工作经验|语言要求|语言|背景要求|文书|推荐信)[：:\s]+")
PREFIX_RE = re.compile(r"^[^：:]{1,40}[：:]")
KEY_TOPIC_WORDS = (
    "topik", "ielts", "toefl", "toeic", "opic", "teps", "雅思", "托福", "语言", "英语", "韩语",
    "gmat", "gre", "文书", "essay", "申请书", "自我介绍", "学业计划", "推荐", "工作经验", "工作经历",
    "专业", "学位", "学历", "毕业", "gpa", "符合", "达到", "满足", "达标", "低于", "高于",
)


def _norm(text: str) -> str:
    """忽略空白、标点和大小写，用于判断原文子串。"""
    return "".join(
        ch for ch in text.lower() if not ch.isspace() and not unicodedata.category(ch)[0] in ("P", "S")
    )


def _flatten(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flatten(v)]
    if isinstance(value, list):
        return [x for v in value for x in _flatten(v)]
    return []


def _all_keys(value: Any) -> set[str]:
    """记录中所有层级的字段名（含 rounds 里的 name、deadline）。"""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _all_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _all_keys(v)}
    return set()


def _record_texts(docs: list[dict[str, Any]]) -> list[str]:
    texts = [t for d in docs for t in _flatten(d)]
    texts += [f"{d.get('school', '')}{d.get('program', '')}" for d in docs]
    return [_norm(t) for t in texts if _norm(t)]


def is_verbatim(piece: str, record_texts: list[str]) -> bool:
    """去掉学校名、"语言："等标签前缀后，是否为学校记录的原文子串（忽略空白和标点）。"""
    core = piece.strip()
    while True:
        stripped = TOPIC_LABEL_RE.sub("", PREFIX_RE.sub("", core, count=1), count=1).strip()
        if stripped == core or not stripped:
            break
        core = stripped
    n = _norm(core)
    return bool(n) and any(n in t for t in record_texts)


def _is_key_statement(piece: str) -> bool:
    lowered = piece.lower()
    return any(w in lowered for w in KEY_TOPIC_WORDS)


def verify_statements(draft: str, docs: list[dict[str, Any]], last_agent: str) -> list[str]:
    """取出需要模型核对的关键陈述：原文子串由代码直接判为有依据，评论性语句不列入。
    选校回答只取各学校的补充说明（符合度由代码判断，不交给模型）。"""
    pieces: list[str] = []
    if last_agent == "school_matcher":
        headers = {school_header(d): d.get("school", "") for d in docs}
        current = ""
        for line in draft.splitlines():
            if line in headers:
                current = headers[line]
            elif line.strip().startswith(SUPPLEMENT_LABEL):
                text = line.strip()[len(SUPPLEMENT_LABEL):]
                pieces += [f"{current}：{p.strip()}" for p in text.split("；") if p.strip()]
    else:
        headers = {f"{d.get('school', '')} {d.get('program', '')}": d.get("school", "") for d in docs}
        current = ""
        for line in draft.splitlines():
            line = line.strip().lstrip("-").strip()
            if line in headers:
                current = headers[line]
                continue
            if not line or line.endswith("：") or line.startswith(STATEMENT_SKIP_PREFIXES):
                continue
            line = ANNOTATION_RE.sub("", line)
            prefix = f"{current}：" if current and not line.startswith(current) else ""
            pieces += [f"{prefix}{p.strip()}" for p in line.split("；") if p.strip()]

    record_texts = _record_texts(docs)
    return [p for p in pieces if _is_key_statement(p) and not is_verbatim(p, record_texts)]


def evidence_parts(evidence: str, field_names: set[str]) -> list[str]:
    """去掉依据里的字段名前缀（如 language:、"background_req": 、major：）和 JSON 标点、引号，
    按字段名拆成若干段原文。"""
    names = "|".join(sorted((re.escape(n) for n in field_names), key=len, reverse=True))
    parts = re.split(rf"[\s{{}}\[\],\"']*\b(?:{names})\b[\s\"']*[:：]", evidence) if names else [evidence]
    return [p.strip(" \t\"'{}[],") for p in parts if p.strip(" \t\"'{}[],")]


def llm_check(
    llm: Any, statements: list[str], docs: list[dict[str, Any]], profile: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """模型为每条关键陈述摘录依据原文，由代码检查依据是否真实出现在学校记录或用户画像中，
    以此决定是否有依据。返回 (不通过的原因, 没有依据的陈述)。调用或解析失败一律视为不通过。"""
    schools = json.dumps([{k: v for k, v in d.items() if k != "aliases"} for d in docs], ensure_ascii=False, indent=1)
    user = json.dumps({f: profile.get(f) for f in PROFILE_FIELDS if profile.get(f)}, ensure_ascii=False, indent=1)
    numbered = "\n".join(f"{i}. {st}" for i, st in enumerate(statements, 1))
    messages = [
        SystemMessage(content=VERIFY_SYSTEM_PROMPT),
        HumanMessage(content=f"【学校记录】\n{schools}\n\n【用户画像】\n{user}\n\n【待核对陈述】\n{numbered}"),
    ]
    try:
        result = llm.with_structured_output(VerifyResult, method="json_schema").invoke(messages)
    except OutputParserException:
        return ["llm_parse_error"], []
    except Exception:  # 超时、网络、拒答等；不记录异常内容
        return [LLM_ERROR], []
    if not isinstance(result, VerifyResult):
        return ["llm_parse_error"], []

    # supported 由代码决定：模型给出的每段依据都必须真实出现在学校记录或用户画像中
    user_values = [profile.get(f) or "" for f in PROFILE_FIELDS]  # 与发给模型的画像字段一致
    source_texts = _record_texts(docs) + [_norm(v) for v in user_values if _norm(v)]
    field_names = _all_keys(docs) | set(PROFILE_FIELDS)
    failed = []
    for c in result.checks:
        if not 1 <= c.index <= len(statements):
            continue
        parts = [p for e in c.evidence for p in evidence_parts(e, field_names)]
        if not parts:
            failed.append((c.index, "未给出依据"))
            continue
        missing = next((p for p in parts if not any(_norm(p) in t for t in source_texts)), None)
        if missing is not None:
            failed.append((c.index, f"依据不在资料中：{missing}"))
    if not failed:
        return [], []
    reasons = [f"llm_unsupported_claims({len(failed)})"] + [
        f"unsupported:{statements[i - 1][:MAX_CLAIM_CHARS]}｜{why[:MAX_REASON_TRACE_CHARS]}" for i, why in failed
    ]
    return reasons, [statements[i - 1] for i, _ in failed]


def verify(state: AdvisorState) -> dict:
    """检查回答中的事实和出处。

    a. 代码核对（始终执行）：草稿非空、需出处的回答带出处、日期都出现在本次检索记录中、
       否定类说法在资料原文中有依据、没有断言录取概率。
    b. 模型核对（有密钥且有检索记录时执行）：只核对关键陈述（选校回答只取各校补充说明），
       学校记录原文子串由代码直接判为有依据；其余由模型摘录依据原文，代码去掉字段名后检查依据是否真实存在，
       没有依据的条数为 0 才通过。选校回答的补充说明未通过时只删除该校补充说明，不影响符合度部分。
    任一项不通过即 fail，原因以 "verify:<原因>" 记录到 trace。
    """
    draft = state.get("draft", "")
    docs = state.get("docs") or []
    reasons = []
    if not draft:
        reasons.append("empty_draft")
    if state.get("last_agent") in SOURCE_REQUIRED_AGENTS and not state.get("sources"):
        reasons.append("no_sources")
    if unsupported_dates(draft, docs, get_today()):
        reasons.append("dates_not_in_sources")
    if unsupported_negations(draft, docs):
        reasons.append("negation_not_in_sources")
    if admission_claims(draft):
        reasons.append("admission_claim")

    llm_ran = no_claims = False
    notes: list[str] = []  # 只记入 trace、不导致 fail 的条目（被删除的补充说明）
    if not reasons and docs:
        llm = get_llm()
        if llm is not None:
            llm = with_max_tokens(llm, VERIFY_MAX_TOKENS)
            last_agent = state.get("last_agent", "")
            statements = verify_statements(draft, docs, last_agent)
            if statements:
                llm_ran = True
                llm_reasons, failed = llm_check(llm, statements, docs, state.get("profile") or {})
                if last_agent == "school_matcher" and llm_reasons:
                    # 补充说明没通过就不显示；模型出错或解析失败时删除全部补充说明。符合度部分不受影响。
                    names = {st.split("：", 1)[0] for st in failed} if failed else None
                    draft = remove_supplements(draft, docs, names)
                    notes = llm_reasons + [f"supplement_removed({n})" for n in (sorted(names) if names else ["all"])]
                else:
                    reasons += llm_reasons
            else:
                no_claims = True  # 关键陈述均为原文子串，无需模型核对

    trace = _trace(state, "verify") + [f"verify:{r}" for r in reasons + notes]
    if llm_ran and not reasons and not notes:
        trace.append("verify:llm_pass")
    if no_claims and not reasons:
        trace.append("verify:no_claims")
    update = {"verdict": "fail" if reasons else "pass", "trace": trace}
    if not reasons:
        update["draft"] = draft
        update["messages"] = [AIMessage(content=draft)]
    return update


def fallback(state: AdvisorState) -> dict:
    """处理无法确认的问题：如实告知，不编造。"""
    reply = (
        "抱歉，这个问题我暂时无法从可靠来源确认。"
        "建议直接查阅目标学校项目的官方网站或联系招生办公室，以获取最新、准确的信息。"
    )
    return {
        "draft": reply,
        "last_agent": "fallback",
        "messages": [AIMessage(content=reply)],
        "trace": _trace(state, "fallback"),
    }


# ---------------------------------------------------------------------------
# Conditional edges（4 条）
# ---------------------------------------------------------------------------

def route_by_intent(state: AdvisorState) -> str:
    """router 之后：按 intent 分流。"""
    return {
        "profile": "profile_extractor",
        "qa": "requirement_retriever",
        "timeline": "requirement_retriever",
        "review": "doc_reviewer",
        "chitchat": END,
    }.get(state.get("intent", ""), "clarify")


def route_after_profile(state: AdvisorState) -> str:
    """profile_extractor 之后：信息完整则选校，否则追问。"""
    return "clarify" if state.get("missing_fields") else "school_matcher"


def route_after_retrieval(state: AdvisorState) -> str:
    """requirement_retriever 之后：无出处则 fallback；有出处按 intent 分流。"""
    if not state.get("sources"):
        return "fallback"
    return "timeline_planner" if state.get("intent") == "timeline" else "verify"


RETRYABLE_AGENTS = {"requirement_retriever", "timeline_planner"}


def _latest_run(trace: list[str], node: str) -> list[str]:
    """trace 中某节点最近一次运行之后的条目。"""
    idx = max((i for i, t in enumerate(trace) if t == node), default=-1)
    return trace[idx + 1:]


def route_after_verify(state: AdvisorState) -> str:
    """verify 之后：通过则结束；模型调用出错直接 fallback；
    问答和时间规划失败且未超过重试上限则回到 last_agent；其余（含选校回答）fallback。"""
    if state.get("verdict") == "pass":
        return END
    if f"verify:{LLM_ERROR}" in _latest_run(state.get("trace", []), "verify"):
        return "fallback"
    if state.get("retry_count", 0) < MAX_RETRIES and state.get("last_agent") in RETRYABLE_AGENTS:
        return state["last_agent"]
    return "fallback"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AdvisorState)

    builder.add_node("router", router)
    builder.add_node("profile_extractor", profile_extractor)
    builder.add_node("clarify", clarify)
    builder.add_node("school_matcher", school_matcher)
    builder.add_node("requirement_retriever", requirement_retriever)
    builder.add_node("doc_reviewer", doc_reviewer)
    builder.add_node("timeline_planner", timeline_planner)
    builder.add_node("verify", verify)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_by_intent,
        ["profile_extractor", "requirement_retriever", "doc_reviewer", "clarify", END],
    )
    builder.add_conditional_edges(
        "profile_extractor",
        route_after_profile,
        ["school_matcher", "clarify"],
    )
    builder.add_conditional_edges(
        "requirement_retriever",
        route_after_retrieval,
        ["verify", "timeline_planner", "fallback"],
    )
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        ["school_matcher", "requirement_retriever", "timeline_planner", "fallback", END],
    )

    builder.add_edge("school_matcher", "verify")
    builder.add_edge("timeline_planner", "verify")
    builder.add_edge("doc_reviewer", END)
    builder.add_edge("clarify", END)
    builder.add_edge("fallback", END)

    return builder.compile()


graph = build_graph()


def run(question: str, profile: Optional[dict] = None) -> AdvisorState:
    state: AdvisorState = {"messages": [HumanMessage(content=question)]}
    if profile:
        state["profile"] = profile
    return graph.invoke(state)


if __name__ == "__main__":
    from config import api_key_hint

    print(api_key_hint())  # 只显示是否已配置，不显示密钥
    nodes =[n for n in graph.get_graph().nodes if n not in ("__start__", "__end__")]
    print(f"节点数量: {len(nodes)} -> {nodes}\n")

    examples = [
        "你好",
        "我想选校，GPA 3.6，本科专业金融，雅思 7.0，GMAT 700，想去英国和香港",
        "我想选校，GPA 3.5",
        "成均馆大学的申请要求是什么？",
        "伦敦商学院的申请要求是什么？",
        "帮我做一个国民大学的申请时间规划",
        "SKK GSB 的申请时间规划",
        "仁济大学和成均馆大学需要 GMAT 吗？",
        "成均馆大学需要工作经验吗？",
        "帮我看看这篇文书：I want to study business analytics because ...",
        "嗯……",
    ]
    for q in examples:
        result = run(q)
        print(f"Q: {q}")
        print(f"  intent : {result.get('intent')}")
        print(f"  trace  : {' -> '.join(result['trace'])}")
        print(f"  answer : {result['messages'][-1].content}\n")
