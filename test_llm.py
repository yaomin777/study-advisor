"""llm.py 测试：只构造客户端、检查请求参数，不发送任何请求。"""

import streamlit as st
from langchain_core.messages import HumanMessage

import config
import llm

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-456"


def test_no_key_returns_none():
    assert llm.get_llm() is None


def test_client_settings(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    model = llm.get_llm()
    assert model is not None
    assert model.model == llm.MODEL_NAME
    assert model.max_retries == 1
    assert model.default_request_timeout == 30
    assert model.temperature is None
    assert model.max_tokens == 1024
    assert model.thinking == {"type": "disabled"}
    assert model.betas is None


def test_reads_streamlit_secrets(monkeypatch):
    monkeypatch.setattr(st, "secrets", {config.API_KEY_NAME: FAKE_KEY})
    assert llm.get_llm() is not None


def test_request_payload_has_no_sampling_params(monkeypatch):
    # API 版本 2026-10-21 下 temperature / top_p / top_k 会导致 400
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    payload = llm.get_llm()._get_request_payload([HumanMessage(content="hi")])
    assert not {"temperature", "top_p", "top_k"} & set(payload)
    assert payload["model"] == llm.MODEL_NAME == "claude-haiku-5-5"
    assert payload["max_tokens"] == 1024
    assert "fallbacks" not in payload  # 已删除拒答换模型重跑


def test_key_not_in_repr(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    model = llm.get_llm()
    assert FAKE_KEY not in repr(model)
    assert FAKE_KEY not in str(model)
    assert FAKE_KEY not in str(model.model_dump())


def test_with_max_tokens_copies_client(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    model = llm.get_llm()
    bigger = llm.with_max_tokens(model, llm.VERIFY_MAX_TOKENS)
    assert llm.VERIFY_MAX_TOKENS == 2048
    assert bigger._get_request_payload([HumanMessage(content="hi")])["max_tokens"] == 2048
    assert model.max_tokens == 1024  # 其他节点不受影响
    assert bigger.thinking == {"type": "disabled"} and bigger.model == llm.MODEL_NAME


def test_with_max_tokens_leaves_fakes_unchanged():
    fake = object()
    assert llm.with_max_tokens(fake, 2048) is fake
