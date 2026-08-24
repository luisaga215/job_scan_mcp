import functools
import logging
import traceback
from typing import Callable, Any, Dict

logger = logging.getLogger(__name__)

def get_error_suggestion(error_type: str, message: str) -> str:
    """Provide actionable suggestions based on the type and message of the error."""
    msg = message.lower()
    if "api_key" in msg or "api key" in msg or "unauthorized" in msg or "401" in msg:
        return "Ensure your API keys (e.g., OPENAI_API_KEY or GEMINI_API_KEY) are correctly configured in your .env file or environment variables."
    if "connection" in msg or "refused" in msg or "11434" in msg:
        return "Connection failed. If using a local model, verify that Ollama is currently running and accessible at the specified OLLAMA_BASE_URL (default: http://localhost:11434)."
    if "not found" in msg or "no such file" in msg:
        return "The requested file could not be found. Double check that the absolute file path is correct and readable."
    if "user profile" in msg or "sync_cv" in msg:
        return "User profile not found. You must successfully run the sync_cv tool with a valid CV file first to populate your profile."
    if "rate limit" in msg or "429" in msg:
        return "Rate limit exceeded. Please wait a short duration before retrying the operation."
    return "Verify the tool inputs are formatted correctly and the configured LLM provider and models are available."

def handle_mcp_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to intercept exceptions in MCP tools and output clean, structured JSON errors."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_type = e.__class__.__name__
            error_msg = str(e)
            logger.error(
                f"Exception in tool '{func.__name__}': {error_type}: {error_msg}\n"
                f"{traceback.format_exc()}"
            )
            return {
                "status": "error",
                "error_type": error_type,
                "message": error_msg,
                "suggestion": get_error_suggestion(error_type, error_msg)
            }
    return wrapper
