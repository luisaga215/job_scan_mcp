import contextlib
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from job_scan_mcp.config import DB_URL, ensure_directories_exist

class DatabaseManager:
    def __init__(self, db_url: str):
        self.db_url = db_url
        # Disable pool pre-ping for SQLite to avoid overhead, enable standard settings
        self.engine = create_async_engine(db_url, echo=False, future=True)
        self.session_maker = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create tables in the database if they do not exist and run lightweight migrations."""
        # For in-memory SQLite, we don't need to create local directories
        if "sqlite+aiosqlite:///:memory:" not in self.db_url:
            ensure_directories_exist()
        
        async with self.engine.begin() as conn:
            # Import models to ensure they are registered with SQLModel's metadata
            from job_scan_mcp.models import LLMConfig, UserProfile, Job, PipelineMeta
            await conn.run_sync(SQLModel.metadata.create_all)
            await conn.run_sync(self._migrate_jobs_table)

    @staticmethod
    def _migrate_jobs_table(sync_conn) -> None:
        """Add columns introduced after the schema was first created (SQLite ALTER TABLE)."""
        from sqlalchemy import text
        existing = {row[1] for row in sync_conn.execute(text("PRAGMA table_info(jobs)")).fetchall()}
        wanted = {
            "sponsorship": "BOOLEAN",
            "relocation_support": "BOOLEAN",
            "us_eligible": "BOOLEAN",
            "visa_keywords": "VARCHAR",
            "application_status": "VARCHAR",
            "application_status_updated_at": "DATETIME",
        }
        for column, ctype in wanted.items():
            if column not in existing:
                sync_conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {column} {ctype}"))

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional async session."""
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Dispose of the engine connection pool."""
        await self.engine.dispose()

# Default global instance
db_manager = DatabaseManager(DB_URL)
