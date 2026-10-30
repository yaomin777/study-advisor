"""AI经营学硕士留学申请顾问 —— Streamlit 界面。

运行：.venv/bin/streamlit run app.py
对话历史与画像保存在 st.session_state 中（不使用 checkpointer）。
"""

import logging
import re

import streamlit as st
from langchain_core.messages import HumanMessage

import graph as g
from config import get_api_key
from llm import MODEL_NAME

MAX_QUESTIONS = 20
DISCLAIMER = "本系统的建议仅供参考，请以学校最新招生简章为准；请勿输入真实个人信息（姓名、证件号、联系方式等）。"
DEMO_MODE = "演示模式（规则版）"
FRIENDLY_ERROR = "抱歉，处理这个问题时出现了错误，请稍后再试或换一种问法。"
LIMIT_REACHED = f"本会话已达到 {MAX_QUESTIONS} 个问题的上限，请刷新页面开始新的会话。"

logger = logging.getLogger(__name__)

# 网址只取 ASCII 可见字符，紧跟其后的中文括号、全角逗号等不会被算进链接
URL_RE = re.compile(r"https?://[!-~]+")


def md_link(url: str, text: str = "") -> str:
    """Markdown 链接 [显示文字](纯网址)；显示文字中的 Markdown 符号加转义。"""
    shown = re.sub(r"([\\`*_\[\]<>])", r"\\\1", text or url)
    return f"[{shown}]({url})"


def link_urls(text: str) -> str:
    """把文本中的裸网址改成 Markdown 链接，网址之后的文字（如"（截至 …）"）留在链接之外。"""
    def repl(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:!?)]'\"")  # 句末 ASCII 标点不算网址
        return md_link(url) + m.group(0)[len(url):]

    return URL_RE.sub(repl, text)


def format_answer(text: str) -> str:
    """显示用：保留回答中的单个换行。缩进的行（各字段）转成列表项，其余各行用硬换行分开；
    不改变回答内容本身。"""
    blocks = []
    for para in text.split("\n\n"):
        out: list[str] = []
        in_list = False
        for line in para.splitlines():
            if not line.strip():
                continue
            if line[:1].isspace():
                out.append(f"- {link_urls(line.strip())}")
                in_list = True
            else:
                if in_list:
                    out.append("")  # 列表后空一行，避免下一行被并入上一个列表项
                    in_list = False
                elif out:
                    out[-1] += "  "  # 硬换行
                out.append(link_urls(line.strip()))
        blocks.append("\n".join(out))
    return "\n\n".join(blocks)


@st.cache_resource
def get_graph():
    return g.build_graph()


@st.cache_resource
def get_schools() -> list[dict]:
    return g.load_schools()


def init_session() -> None:
    st.session_state.setdefault("history", [])  # [{"role": "user"|"assistant", "content": str}]
    st.session_state.setdefault("profile", {})
    st.session_state.setdefault("question_count", 0)
    st.session_state.setdefault("last_result", None)


def mode_label() -> str:
    return f"模型模式（{MODEL_NAME}）" if get_api_key() else DEMO_MODE


def ask(question: str) -> str:
    """运行一轮对话，返回回答；出错时返回友好提示，不显示堆栈。"""
    state = {"messages": [HumanMessage(content=question)], "profile": dict(st.session_state.profile)}
    try:
        result = get_graph().invoke(state, config={"configurable": {"schools": get_schools()}})
    except Exception as e:
        # 只记录异常类型，异常内容可能包含请求细节
        logger.warning("graph invoke failed: %s", type(e).__name__)
        st.session_state.last_result = None
        return FRIENDLY_ERROR
    st.session_state.profile = result.get("profile") or st.session_state.profile
    st.session_state.last_result = result
    return result["messages"][-1].content


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"**运行模式：** {mode_label()}")
        st.markdown(f"**已提问：** {st.session_state.question_count} / {MAX_QUESTIONS}")
        dates = sorted({s["as_of"] for s in get_schools() if s.get("as_of")})
        st.markdown(f"**数据整理日期：** {'、'.join(dates) or '未知'}")

        result = st.session_state.last_result
        st.subheader("本轮节点路径")
        if result:
            trace = result.get("trace", [])
            nodes = [t for t in trace if ":" not in t]
            st.markdown("\n".join(f"{i}. `{t}`" for i, t in enumerate(nodes, 1)))  # 纵向列表，不被裁切
            with st.expander("完整 trace"):
                st.code(" → ".join(trace), language=None, wrap_lines=True)
        else:
            st.caption("尚无")

        st.subheader("当前画像")
        if st.session_state.profile:
            st.json(st.session_state.profile)
        else:
            st.caption("尚未提取")

        st.subheader("检索到的出处")
        docs = (result or {}).get("docs") or []
        if docs:
            for d in docs:
                url = d.get("source_url", "")
                link = md_link(url, d.get("school", "")) if URL_RE.fullmatch(url) else d.get("school", "")
                st.markdown(f"- {link}（整理日期 {d.get('as_of', '未知')}）")
        else:
            st.caption("本轮无")


def main() -> None:
    st.set_page_config(page_title="经营学硕士申请顾问", page_icon="🎓")
    st.title("AI 经营学硕士留学申请顾问")
    st.warning(DISCLAIMER)
    init_session()

    if not get_api_key():
        st.info(f"当前为{DEMO_MODE}：未配置 ANTHROPIC_API_KEY，回答由规则生成。")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(format_answer(msg["content"]) if msg["role"] == "assistant" else msg["content"])

    limit_reached = st.session_state.question_count >= MAX_QUESTIONS
    if limit_reached:
        st.warning(LIMIT_REACHED)
    question = st.chat_input("请输入你的问题，例如：成均馆大学需要 GMAT 吗？", disabled=limit_reached)

    if question and not limit_reached:
        st.session_state.question_count += 1
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                answer = ask(question)
            st.markdown(format_answer(answer))
        st.session_state.history.append({"role": "assistant", "content": answer})

    render_sidebar()


try:
    main()
except Exception as e:  # 兜底：任何渲染错误都只显示友好提示
    logger.warning("app error: %s", type(e).__name__)
    st.error(FRIENDLY_ERROR)
