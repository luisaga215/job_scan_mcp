import pytest
from unittest.mock import patch
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from job_scan_mcp.services import llm_factory
from job_scan_mcp.services.llm_factory import (
    parse_model_string,
    create_llm,
    get_llm_config_for_stage,
    JSONStructuredLLM,
    DeepSeekChatLLM,
)


class _SampleSchema(BaseModel):
    is_relevant: bool
    reason: str


class _FakeLLM:
    async def ainvoke(self, prompt, *args, **kwargs):
        return type("Response", (), {"content": 'Here: {"is_relevant": true, "reason": "great fit"}'})()


class _BadLLM:
    async def ainvoke(self, prompt, *args, **kwargs):
        return type("Response", (), {"content": "no json here"})()


def test_parse_model_string_deepseek():
    assert parse_model_string("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")
    assert parse_model_string("gemini/gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")


def test_create_llm_deepseek():
    with patch.object(llm_factory.config, "DEEPSEEK_API_KEY", "sk-test"):
        llm = create_llm("deepseek", "deepseek-chat", base_url="https://api.deepseek.com")
    assert isinstance(llm, ChatOpenAI)
    assert isinstance(llm, DeepSeekChatLLM)
    assert llm.model_name == "deepseek-chat"


def test_deepseek_structured_output_uses_json_fallback():
    with patch.object(llm_factory.config, "DEEPSEEK_API_KEY", "sk-test"):
        llm = create_llm("deepseek", "deepseek-chat")
    wrapped = llm.with_structured_output(_SampleSchema)
    assert isinstance(wrapped, JSONStructuredLLM)


@pytest.mark.asyncio
async def test_json_structured_llm_parses_and_validates():
    wrapper = JSONStructuredLLM(_FakeLLM(), _SampleSchema)
    result = await wrapper.ainvoke("prompt")
    assert result.is_relevant is True
    assert result.reason == "great fit"


@pytest.mark.asyncio
async def test_json_structured_llm_raises_when_json_missing():
    wrapper = JSONStructuredLLM(_BadLLM(), _SampleSchema)
    with pytest.raises(ValueError, match="Could not extract a JSON object"):
        await wrapper.ainvoke("prompt")


def test_create_llm_deepseek_requires_key():
    with patch.object(llm_factory.config, "DEEPSEEK_API_KEY", None):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            create_llm("deepseek", "deepseek-chat")


@pytest.mark.asyncio
async def test_get_llm_config_for_stage_deepseek_default(test_repo):
    with patch.object(llm_factory.config, "DEFAULT_SCREENING_MODEL", "deepseek/deepseek-chat"), \
         patch.object(llm_factory.config, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"):
        provider, model, base_url = await get_llm_config_for_stage("fast_screening", test_repo)
    assert provider == "deepseek"
    assert model == "deepseek-chat"
    assert base_url == "https://api.deepseek.com"
