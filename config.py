"""模型密钥配置。

读取顺序：系统环境变量 → 本地 .env → Streamlit 部署时的 st.secrets。
任何函数都不打印、不记录、不返回密钥明文以外的信息到界面或日志。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

API_KEY_NAME = "ANTHROPIC_API_KEY"
ENV_PATH = Path(__file__).parent / ".env"

MISSING_KEY_MESSAGE = (
    f"未检测到模型密钥 {API_KEY_NAME}。"
    f"本地运行请复制 .env.example 为 .env 并填写 {API_KEY_NAME}；"
    f"部署到 Streamlit 时请在 secrets（.streamlit/secrets.toml 或云端 Secrets 设置）中配置 {API_KEY_NAME}。"
    "当前使用规则版（verify 只做代码核对，不调用模型）。"
)


def _from_env_file(name: str) -> Optional[str]:
    if not ENV_PATH.exists():
        return None
    # 只读取文件，不写入 os.environ，避免影响其他进程或测试
    return dotenv_values(ENV_PATH).get(name) or None


def _from_streamlit_secrets(name: str) -> Optional[str]:
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:  # 未安装 streamlit、没有 secrets.toml 等
        return None
    return str(value) if value else None


def get_api_key(name: str = API_KEY_NAME) -> Optional[str]:
    """返回密钥；都没有配置时返回 None。调用方不得打印返回值。"""
    return os.getenv(name) or _from_env_file(name) or _from_streamlit_secrets(name)


def api_key_hint() -> str:
    """给界面显示的状态说明，不包含密钥内容。"""
    return "模型密钥：已配置" if get_api_key() else MISSING_KEY_MESSAGE
