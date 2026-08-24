import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

from job_scan_mcp.database import db_manager
from job_scan_mcp import config
from job_scan_mcp.repository import JobRepository
from job_scan_mcp.services.cv_service import parse_and_sync_cv
from job_scan_mcp.services.job_service import fetch_and_save_jobs
from job_scan_mcp.services.screening import run_fast_screening_batch
from job_scan_mcp.services.evaluation import run_deep_evaluation_batch
from job_scan_mcp.services.report import generate_report
from job_scan_mcp.utils.errors import handle_mcp_errors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("job-scan-mcp")

async def ensure_db() -> None:
    """Ensure tables and folder structure are initialized."""
    # This is also run during lazy initialization if main wasn't executed (e.g. testing)
    await db_manager.init_db()

@mcp.tool()
@handle_mcp_errors
async def configure_llm(stage: str, provider: str, model: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Dynamically configure LLM provider settings for specific processing stages.

    Args:
        stage: Pipeline stage to configure. Must be 'fast_screening' or 'deep_evaluation'.
        provider: AI Provider. Must be 'openai', 'ollama', or 'gemini'.
        model: Model name/identifier (e.g., 'gpt-4o-mini', 'llama3', 'gemini-1.5-flash').
        base_url: Optional custom server endpoint URL for OpenAI/Ollama providers.
    """
    if stage not in ["fast_screening", "deep_evaluation"]:
        raise ValueError("The 'stage' argument must be either 'fast_screening' or 'deep_evaluation'.")
        
    provider = provider.lower()
    if provider not in ["openai", "ollama", "gemini", "deepseek"]:
        raise ValueError("The 'provider' argument must be one of: 'openai', 'ollama', 'gemini', 'deepseek'.")
        
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        config_record = await repo.save_llm_config(
            stage=stage,
            provider=provider,
            model=model,
            base_url=base_url
        )
        return {
            "status": "success",
            "config": {
                "stage": config_record.stage,
                "provider": config_record.provider,
                "model": config_record.model,
                "base_url": config_record.base_url,
                "updated_at": config_record.updated_at.isoformat()
            }
        }

@mcp.tool()
@handle_mcp_errors
async def sync_cv(file_path: str) -> Dict[str, Any]:
    """Parse a CV file (PDF or Markdown), extract candidate details, and save the profile.

    Args:
        file_path: The absolute local file path to the PDF, MD, or TXT resume.
    """
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        user_profile = await parse_and_sync_cv(file_path, repo)
        return {
            "status": "success",
            "message": "User CV successfully parsed and profile synced in the database.",
            "profile": user_profile.profile.model_dump()
        }

@mcp.tool()
@handle_mcp_errors
async def get_user_profile() -> Dict[str, Any]:
    """Fetch the active standardized candidate profile saved in the database."""
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        user_profile = await repo.get_user_profile()
        if not user_profile:
            return {
                "status": "error",
                "message": "No profile synced yet. Please run sync_cv first."
            }
        return {
            "status": "success",
            "profile": user_profile.profile.model_dump()
        }

@mcp.tool()
@handle_mcp_errors
async def fetch_and_filter_jobs(
    queries: List[str],
    locations: List[str],
    max_pool_size: int = 150,
    is_remote: Optional[bool] = None,
    min_salary: Optional[float] = None,
    require_visa_friendly: Optional[bool] = None
) -> Dict[str, Any]:
    """Scrape jobs via jobspy, apply deterministic Python filters, and save new postings as PENDING_SCREENING.

    Args:
        queries: List of job roles to search (e.g. ["Software Engineer", "Python Developer"]).
        locations: List of regions to scan (e.g. ["Austin, TX", "Remote"]).
        max_pool_size: Maximum total jobs to query across combinations (default: 150).
        is_remote: Optional filter. True: Remote-only, False: Onsite/Hybrid, None: All.
        min_salary: Optional minimum annual salary threshold (tolerates None fields).
        require_visa_friendly: Optional strict filter. True: keep ONLY postings that explicitly offer visa sponsorship or relocation.
    """
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        res = await fetch_and_save_jobs(
            repo=repo,
            queries=queries,
            locations=locations,
            max_pool_size=max_pool_size,
            is_remote=is_remote,
            min_salary=min_salary,
            require_visa_friendly=require_visa_friendly
        )
        return {
            "status": "success",
            "data": res
        }

@mcp.tool()
@handle_mcp_errors
async def run_fast_screening(batch_size: int = 50, time_budget_seconds: Optional[float] = None, concurrency: Optional[int] = None) -> Dict[str, Any]:
    """Perform quick relevance screening (PENDING_SCREENING -> RELEVANT or REJECTED) on a batch of jobs.

    Args:
        batch_size: Number of jobs to process in this run (default: 50).
        time_budget_seconds: Optional max wall-clock seconds before returning partial results (default: 25).
        concurrency: Optional max concurrent LLM calls (default: 3).
    """
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        res = await run_fast_screening_batch(batch_size, repo, time_budget_seconds=time_budget_seconds, concurrency=concurrency)
        return {
            "status": "success",
            "data": res
        }

@mcp.tool()
@handle_mcp_errors
async def run_deep_evaluation(batch_size: int = 15, time_budget_seconds: Optional[float] = None, concurrency: Optional[int] = None) -> Dict[str, Any]:
    """Perform LangGraph deep evaluation (RELEVANT -> EVALUATED) extracting seniority fit and red flags.

    Args:
        batch_size: Number of jobs to process in this run (default: 15).
        time_budget_seconds: Optional max wall-clock seconds before returning partial results (default: 25).
        concurrency: Optional max concurrent LLM calls (default: 3).
    """
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        res = await run_deep_evaluation_batch(batch_size, repo, time_budget_seconds=time_budget_seconds, concurrency=concurrency)
        return {
            "status": "success",
            "data": res
        }

@mcp.tool()
@handle_mcp_errors
async def get_pipeline_status() -> Dict[str, Any]:
    """Fetch pipeline operational statistics, state counts, and active LLM configuration mappings."""
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        counts = await repo.get_pipeline_counts()
        fast_cfg = await repo.get_llm_config("fast_screening")
        deep_cfg = await repo.get_llm_config("deep_evaluation")
        profile = await repo.get_user_profile()
        
        return {
            "status": "success",
            "pipeline_counts": counts,
            "llm_configurations": {
                "fast_screening": {
                    "provider": fast_cfg.provider if fast_cfg else config.DEFAULT_SCREENING_MODEL.split("/")[0],
                    "model": fast_cfg.model if fast_cfg else config.DEFAULT_SCREENING_MODEL,
                    "base_url": fast_cfg.base_url if fast_cfg else None
                },
                "deep_evaluation": {
                    "provider": deep_cfg.provider if deep_cfg else config.DEFAULT_EVALUATION_MODEL.split("/")[0],
                    "model": deep_cfg.model if deep_cfg else config.DEFAULT_EVALUATION_MODEL,
                    "base_url": deep_cfg.base_url if deep_cfg else None
                }
            },
            "user_profile_synced": profile is not None
        }

@mcp.tool()
@handle_mcp_errors
async def set_job_application_status(job_id: str, status: str) -> Dict[str, Any]:
    """Persist the user-managed application status for a job (kanban workflow).

    Args:
        job_id: Deterministic job id (SHA-256 hash) shown in the HTML report.
        status: One of 'apply', 'applied', 'interview', 'rejected'.
    """
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        job = await repo.set_application_status(job_id, status)
        if job is None:
            return {
                "status": "error",
                "message": f"No job found with id '{job_id}'."
            }
        return {
            "status": "success",
            "message": "Application status updated.",
            "job": {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "application_status": job.application_status
            }
        }

@mcp.tool()
@handle_mcp_errors
async def archive_report(file: str) -> Dict[str, Any]:
    """Move a report snapshot to the archive folder (soft delete, reversible). The report stays on disk in reports/archive.

    Args:
        file: Snapshot filename from the report history (e.g. 'report-20260823-120000-000000.html').
    """
    from job_scan_mcp.services.report import archive_report_file
    result = archive_report_file(file)
    if not result.get("ok"):
        return {"status": "error", "message": result.get("error", "Archive failed.")}
    return {"status": "success", "data": result}

@mcp.tool()
@handle_mcp_errors
async def restore_report(file: str) -> Dict[str, Any]:
    """Move a report snapshot back from the archive folder to the active reports directory.

    Args:
        file: Snapshot filename currently in the archive (e.g. 'report-20260823-120000-000000.html').
    """
    from job_scan_mcp.services.report import restore_report_file
    result = restore_report_file(file)
    if not result.get("ok"):
        return {"status": "error", "message": result.get("error", "Restore failed.")}
    return {"status": "success", "data": result}

@mcp.tool()
@handle_mcp_errors
async def generate_tailored_cv(job_description_text: str, base_cv_json: dict) -> Dict[str, Any]:
    """Analyze a job description and rewrite the base CV to maximize alignment (Dynamic CV Tailoring).

    Returns a JSON CV where every modified bullet carries {'modified': true, 'match_reason': '...'},
    and writes a self-contained preview HTML with the highlight engine for review.

    Args:
        job_description_text: The full job description text.
        base_cv_json: The candidate's base CV as a JSON structure (name, contact, summary, experience, education, skills).
    """
    await ensure_db()
    from job_scan_mcp.services import cv_tailor
    async with db_manager.session() as session:
        repo = JobRepository(session)
        tailored = await cv_tailor.generate_tailored_cv(job_description_text, base_cv_json, repo)
        preview = cv_tailor.render_preview_html(tailored)
        return {
            "status": "success",
            "data": {
                "tailored_cv": tailored,
                "modified_bullets": tailored.get("_meta", {}).get("modified_bullets", 0),
                "total_bullets": tailored.get("_meta", {}).get("total_bullets", 0),
                "preview_html": preview.as_uri(),
            }
        }

@mcp.tool()
@handle_mcp_errors
async def export_cv_to_pdf(tailored_cv_data: dict, file_name: str) -> Dict[str, Any]:
    """Render the approved tailored CV JSON to an ATS-friendly PDF (Playwright).

    Args:
        tailored_cv_data: The tailored CV JSON (output of generate_tailored_cv).
        file_name: Desired PDF file name (extension optional).
    """
    await ensure_db()
    from job_scan_mcp.services import cv_tailor
    pdf_path = await cv_tailor.export_cv_to_pdf(tailored_cv_data, file_name)
    return {
        "status": "success",
        "data": {
            "pdf_path": pdf_path,
            "pdf_uri": Path(pdf_path).as_uri(),
        }
    }

@mcp.tool()
@handle_mcp_errors
async def generate_html_report() -> Dict[str, Any]:
    """Compile database jobs into an interactive HTML dashboard report, saving it locally."""
    await ensure_db()
    async with db_manager.session() as session:
        repo = JobRepository(session)
        report_uri = await generate_report(repo)
        return {
            "status": "success",
            "report_path": report_uri
        }

def main() -> None:
    """Entry point for the console script."""
    # Force DB initial schema generation synchronously before startup
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(db_manager.init_db())
    except Exception as e:
        logger.error(f"Auto DB initialization failed on startup: {str(e)}")
        
    mcp.run()

if __name__ == "__main__":
    main()
