import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file.
# Load from the current working directory first (takes precedence), then fall back
# to the project root so API keys are found when clients launch the server elsewhere.
load_dotenv()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# Base directories
DEFAULT_DATA_DIR = Path.home() / ".job_evaluator"
DATA_DIR_ENV = os.getenv("DATA_DIR")

if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve()
else:
    DATA_DIR = DEFAULT_DATA_DIR.expanduser().resolve()

# Subdirectories
REPORTS_DIR = DATA_DIR / "reports"

# Database Configuration
DB_PATH = DATA_DIR / "jobs.db"
# Use aiosqlite for async SQLite support
DB_URL = f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"

# API Keys & Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Defaults
DEFAULT_SCREENING_MODEL = os.getenv("DEFAULT_SCREENING_MODEL", "openai/gpt-4o-mini")
DEFAULT_EVALUATION_MODEL = os.getenv("DEFAULT_EVALUATION_MODEL", "openai/gpt-4o")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def ensure_directories_exist():
    """Ensure that the local data and reports directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
