"""统一的模型入口。

get_llm() 取不到密钥时返回 None，调用方据此退回规则版。
密钥读取顺序见 config.get_api_key()：环境变量 ANTHROPIC_API_KEY → 本地 .env → st.secrets。
不打印、不记录密钥；ChatAnthropic 内部以 SecretStr 保存，repr 中不会出现明文。
"""

from typing import Any, Optional

from config import get_api_key

MODEL_NAME = "claude-haiku-5-5"
TIMEOUT_SECONDS = 30
MAX_RETRIES = 1
MAX_TOKENS = 1024
VERIFY_MAX_TOKENS = 2048  # verify 需为每条陈述摘录依据，1024 时偶发输出被截断导致解析失败

# 不设置 temperature：当前 SDK（anthropic 2.x）使用 API 版本 2026-10-21，
# 该版本下请求中出现 temperature / top_p / top_k 一律返回 400。
#
# 关闭思考：claude-haiku-5-5 默认开启思考，思考 token 计入 max_tokens，
# 1024 的上限可能被思考占满而截断结构化输出。关闭思考仅在 effort 为 high 及以下时可用（默认 medium）。
THINKING = {"type": "disabled"}


def get_llm() -> Optional[Any]:
    api_key = get_api_key()
    if not api_key:
        return None
    from langchain_anthropic import ChatAnthropic  # 无密钥时不导入

    return ChatAnthropic(
        model=MODEL_NAME,
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        max_tokens=MAX_TOKENS,
        thinking=THINKING,
    )


def with_max_tokens(model: Any, max_tokens: int) -> Any:
    """返回 max_tokens 改为指定值的模型副本。包装对象由其自身的 with_max_tokens 处理；
    没有 max_tokens 设置的对象（测试用假模型等）原样返回。"""
    if hasattr(model, "with_max_tokens"):
        return model.with_max_tokens(max_tokens)
    if hasattr(model, "max_tokens") and hasattr(model, "model_copy"):
        return model.model_copy(update={"max_tokens": max_tokens})
    return model
