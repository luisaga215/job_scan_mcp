import json
import logging
from typing import Any, Optional, Tuple
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Robust ChatOllama import supporting different versions of LangChain
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models.ollama import ChatOllama
        except ImportError:
            class ChatOllama:  # type: ignore
                def __init__(self, *args, **kwargs):
                    raise ImportError(
                        "Ollama driver not found. Please install 'langchain-ollama' or "
                        "'langchain-community' in your environment to use Ollama models."
                    )
from job_scan_mcp import config
from job_scan_mcp.repository import JobRepository

logger = logging.getLogger(__name__)

def parse_model_string(model_str: str) -> Tuple[str, str]:
    """Parse a string like 'openai/gpt-4o-mini' into (provider, model_name)."""
    if "/" in model_str:
        provider, model = model_str.split("/", 1)
        return provider.lower(), model
    return "openai", model_str


class JSONStructuredLLM:
    """Structured-output fallback for providers without response_format support (e.g. DeepSeek).

    Prompts for a single JSON object, extracts it from the raw text, and validates it
    against a Pydantic schema. Exposes the same with_structured_output/ainvoke contract
    that LangChain call sites rely on.
    """

    def __init__(self, llm: Any, schema: type[BaseModel]):
        self.llm = llm
        self.schema = schema

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, prompt, *args, **kwargs):
        schema_json = json.dumps(self.schema.model_json_schema())
        json_prompt = (
            f"{prompt}\n\n"
            "Respond ONLY with a single valid JSON object conforming to this JSON schema "
            "(no markdown fences, no commentary, no extra keys):\n"
            f"{schema_json}"
        )
        raw = await self.llm.ainvoke(json_prompt, *args, **kwargs)
        text = raw.content if hasattr(raw, "content") else str(raw)
        text = str(text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not extract a JSON object from model output: {text[:300]}")
        payload = json.loads(text[start:end + 1])
        return self.schema.model_validate(payload)


class DeepSeekChatLLM(ChatOpenAI):
    """ChatOpenAI wrapper that routes with_structured_output to the JSON fallback.

    DeepSeek's OpenAI-compatible API does not support the response_format parameter
    that langchain-openai sends for structured outputs.
    """

    def with_structured_output(self, schema, **kwargs):
        return JSONStructuredLLM(self, schema)

async def get_llm_config_for_stage(stage: str, repo: JobRepository) -> Tuple[str, str, Optional[str]]:
    """Retrieve LLM config from database if it exists, otherwise fall back to environment defaults."""
    db_config = await repo.get_llm_config(stage)
    if db_config:
        return db_config.provider, db_config.model, db_config.base_url

    # Fallback to defaults loaded from env
    default_model_str = (
        config.DEFAULT_SCREENING_MODEL if stage == "fast_screening"
        else config.DEFAULT_EVALUATION_MODEL
    )
    provider, model = parse_model_string(default_model_str)
    
    # Resolve default base URLs
    base_url = None
    if provider == "openai":
        base_url = config.OPENAI_BASE_URL
    elif provider == "ollama":
        base_url = config.OLLAMA_BASE_URL
    elif provider == "deepseek":
        base_url = config.DEEPSEEK_BASE_URL
        
    return provider, model, base_url

def create_llm(provider: str, model: str, base_url: Optional[str] = None) -> BaseChatModel:
    """Create a LangChain chat model instance based on provider and model parameters."""
    provider = provider.lower()
    
    if provider == "openai":
        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url or config.OPENAI_BASE_URL,
            temperature=0.0
        )
        
    elif provider == "ollama":
        url = base_url or config.OLLAMA_BASE_URL
        return ChatOllama(
            model=model,
            base_url=url,
            temperature=0.0
        )
        
    elif provider == "gemini":
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0
        )
        
    elif provider == "deepseek":
        api_key = config.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set in environment variables.")
        # DeepSeek exposes an OpenAI-compatible endpoint; structured output is handled
        # via the JSON fallback because response_format is not supported.
        return DeepSeekChatLLM(
            model=model,
            api_key=api_key,
            base_url=base_url or config.DEEPSEEK_BASE_URL,
            temperature=0.0
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Supported providers are: openai, ollama, gemini, deepseek.")

async def get_llm_for_stage(stage: str, repo: JobRepository) -> BaseChatModel:
    """Fetch the correct LLM instance for a specific pipeline stage."""
    provider, model, base_url = await get_llm_config_for_stage(stage, repo)
    logger.info(f"Creating LLM for stage '{stage}': provider={provider}, model={model}, base_url={base_url}")
    return create_llm(provider, model, base_url)
