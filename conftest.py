"""pytest 全局设置：屏蔽所有密钥来源，保证测试不会调用真实模型。"""

import pytest
import streamlit as st

import config


@pytest.fixture(autouse=True)
def no_real_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv(config.API_KEY_NAME, raising=False)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(st, "secrets", {})
