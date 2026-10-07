from collections.abc import Iterator

import pytest

from services.config import get_settings
from services.llm import chat_model, code_model


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Lets a test set environment variables; get_settings is re-read before and after."""
    get_settings.cache_clear()
    yield monkeypatch
    monkeypatch.undo()
    get_settings.cache_clear()


def test_code_model_defaults_to_the_general_model(settings_env: pytest.MonkeyPatch):
    settings_env.setenv("LLM_MODEL", "gpt-4.1-mini")
    settings_env.setenv("LLM_CODE_MODEL", "")  # empty, as in .env.example
    assert code_model().model_name == "gpt-4.1-mini"


def test_code_model_can_differ(settings_env: pytest.MonkeyPatch):
    settings_env.setenv("LLM_MODEL", "gpt-4.1-mini")
    settings_env.setenv("LLM_CODE_MODEL", "claude-sonnet-4-5")
    assert chat_model().model_name == "gpt-4.1-mini"
    assert code_model().model_name == "claude-sonnet-4-5"


def test_temperature_none_is_not_sent(settings_env: pytest.MonkeyPatch):
    settings_env.setenv("LLM_CODE_TEMPERATURE", "none")
    settings_env.setenv("LLM_TEMPERATURE", "0")
    # ChatOpenAI leaves a None temperature out of the request
    assert code_model().temperature is None
    assert chat_model().temperature == 0
