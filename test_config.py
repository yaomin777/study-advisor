"""config.py 的密钥读取测试。运行：.venv/bin/python -m pytest -q"""

import subprocess
import sys
from pathlib import Path

import streamlit as st

import config

# 环境隔离（无环境变量、.env 指向临时目录、st.secrets 为空）由 conftest.py 统一处理
FAKE_KEY = "sk-test-DO-NOT-LEAK-123"


class _RaisingSecrets:
    def get(self, name):
        raise FileNotFoundError("no secrets.toml")


def test_from_environment(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    assert config.get_api_key() == FAKE_KEY


def test_from_env_file(tmp_path):
    (tmp_path / ".env").write_text(f"{config.API_KEY_NAME}={FAKE_KEY}\n", encoding="utf-8")
    assert config.get_api_key() == FAKE_KEY


def test_empty_env_file_value(tmp_path):
    (tmp_path / ".env").write_text(f"{config.API_KEY_NAME}=\n", encoding="utf-8")
    assert config.get_api_key() is None


def test_from_streamlit_secrets(monkeypatch):
    monkeypatch.setattr(st, "secrets", {config.API_KEY_NAME: FAKE_KEY})
    assert config.get_api_key() == FAKE_KEY


def test_env_takes_priority_over_secrets(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, "from-env")
    monkeypatch.setattr(st, "secrets", {config.API_KEY_NAME: "from-secrets"})
    assert config.get_api_key() == "from-env"


def test_missing_everywhere_shows_hint(monkeypatch):
    monkeypatch.setattr(st, "secrets", _RaisingSecrets())
    assert config.get_api_key() is None
    assert config.api_key_hint() == config.MISSING_KEY_MESSAGE


def test_hint_does_not_leak_key(monkeypatch):
    monkeypatch.setenv(config.API_KEY_NAME, FAKE_KEY)
    assert FAKE_KEY not in config.api_key_hint()


def test_graph_cli_does_not_print_key():
    # 指向本机未监听的端口：模型调用立即失败，不会访问真实 API，同时覆盖错误路径的输出
    root = Path(__file__).parent
    env = {
        "PATH": "/usr/bin:/bin",
        config.API_KEY_NAME: FAKE_KEY,
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    }
    out = subprocess.run(
        [sys.executable, "graph.py"], cwd=root, env=env, capture_output=True, text=True, check=True,
        timeout=120,
    )
    assert "模型密钥：已配置" in out.stdout
    assert "router -> router:llm_error" in out.stdout
    assert FAKE_KEY not in out.stdout + out.stderr


def test_gitignore_excludes_secrets():
    lines = (Path(__file__).parent / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert ".streamlit/secrets.toml" in lines
