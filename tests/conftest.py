import pytest
import pytest_asyncio
from sqlmodel import SQLModel
from job_scan_mcp.database import DatabaseManager, db_manager
from job_scan_mcp.repository import JobRepository
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

# Redirect global db_manager to in-memory for all tests
db_manager.db_url = "sqlite+aiosqlite:///:memory:"
db_manager.engine = create_async_engine(db_manager.db_url, echo=False, future=True)
db_manager.session_maker = async_sessionmaker(
    bind=db_manager.engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@pytest.fixture(autouse=True)
def _isolate_reports_dir(tmp_path, monkeypatch):
    """Keep every test away from the real reports directory (no manifest/history pollution)."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path / "reports")


@pytest.fixture(autouse=True)
def _isolate_cv_tailor_data(tmp_path, monkeypatch):
    """Keep CV tailoring results/exports away from the real data directory."""
    from job_scan_mcp.services import cv_tailor
    monkeypatch.setattr(cv_tailor.config, "DATA_DIR", tmp_path / "data")


@pytest.fixture(autouse=True)
def _no_real_llm_for_report_summaries(monkeypatch):
    """Reports generate an AI summary; avoid real API calls in tests."""

    class _FakeLLM:
        async def ainvoke(self, prompt):
            return type("Resp", (), {"content": "• Fake summary bullet for tests"})

    async def _fake_get(*args, **kwargs):
        return _FakeLLM()

    monkeypatch.setattr("job_scan_mcp.services.llm_factory.get_llm_for_stage", _fake_get)


class MockStructuredLLM:
    """Mock LangChain chat model wrapper that mocks with_structured_output and ainvoke."""
    def __init__(self, return_value):
        self.return_value = return_value
        self.last_prompt = None

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt, *args, **kwargs):
        self.last_prompt = prompt
        return self.return_value

@pytest_asyncio.fixture
async def test_db():
    """Setup a transient in-memory async SQLite database."""
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await manager.init_db()
    try:
        yield manager
    finally:
        await manager.close()

@pytest_asyncio.fixture
async def test_session(test_db):
    """Provide a transactional test database session."""
    async with test_db.session() as session:
        yield session

@pytest_asyncio.fixture
async def test_repo(test_session):
    """Provide a repository instance bound to the test session."""
    return JobRepository(test_session)
