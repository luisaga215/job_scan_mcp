import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from job_scan_mcp.models import Job
from job_scan_mcp.repository import JobRepository
from job_scan_mcp.services.llm_factory import get_llm_for_stage

logger = logging.getLogger(__name__)

# =====================================================================
# Strict concurrency & resilience controls
# =====================================================================
SCREENING_CONCURRENCY = 3            # Max concurrent LLM calls (avoids rate-limit saturation)
SCREENING_JOB_TIMEOUT_SECONDS = 40   # Hard per-job timeout (prevents hangs blocking the batch)
SCREENING_TIME_BUDGET_SECONDS = 25   # Return partial results after this budget, so the MCP client never times out
SCREENING_CHUNK_SIZE = 10            # Jobs per scheduling wave


class FastScreeningResult(BaseModel):
    is_relevant: bool = Field(..., description="True if the job is a reasonable match for the candidate's skills, experience, and profile. False otherwise.")
    reason: str = Field(..., description="A clear, single-sentence explanation of why the candidate fits or is rejected for this role.")


async def screen_single_job(
    job: Job,
    candidate_profile_summary: str,
    candidate_stack: List[str],
    candidate_skills: List[str],
    candidate_experience: float,
    candidate_seniority: str,
    structured_llm: Any,
    semaphore: asyncio.Semaphore,
    job_timeout: float = SCREENING_JOB_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Classify a single job using the structured LLM with concurrency and timeout gating.

    Performs the LLM call only; DB persistence happens sequentially in the caller
    to avoid concurrent flushes on a shared SQLAlchemy session.
    """
    async with semaphore:
        visa_signals = job.visa_keywords or "none detected"
        relocation = "yes" if job.relocation_support else ("no" if job.relocation_support is False else "unknown")
        prompt = (
            "You are a recruitment screening assistant. Evaluate if the following job posting is relevant to the candidate's profile.\n\n"
            "=== CANDIDATE PROFILE ===\n"
            f"Seniority Level: {candidate_seniority}\n"
            f"Years of Experience: {candidate_experience}\n"
            f"Core Stack: {', '.join(candidate_stack)}\n"
            f"Skills: {', '.join(candidate_skills)}\n"
            f"Summary: {candidate_profile_summary}\n\n"
            "=== JOB DETAILS ===\n"
            f"Title: {job.title}\n"
            f"Company: {job.company}\n"
            f"Location: {job.location}\n"
            f"Visa/Sponsorship signals: {visa_signals}\n"
            f"Relocation support: {relocation}\n"
            f"Description Summary (First 3000 chars):\n{job.description[:3000]}\n\n"
            "Determine if this job is relevant. If the candidate does not have the required experience, "
            "if there is no stack alignment, or if the role doesn't fit the candidate's professional trajectory, mark relevant as false.\n"
            "IMPORTANT: If the job explicitly states the employer will NOT sponsor visas or requires the candidate to "
            "already be authorized to work in the US (e.g., 'must be authorized to work', 'US citizens only', 'no visa "
            "sponsorship'), mark relevant as false because the candidate requires work authorization sponsorship."
        )
        
        try:
            result: FastScreeningResult = await asyncio.wait_for(
                structured_llm.ainvoke(prompt),
                timeout=job_timeout
            )
            
            return {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "is_relevant": result.is_relevant,
                "reason": result.reason
            }
        except asyncio.TimeoutError:
            logger.error(f"Screening job {job.id} ({job.title}) timed out after {job_timeout}s")
            # Do not update the job state, so it can be retried later
            return {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "error": f"LLM call timed out after {job_timeout}s"
            }
        except Exception as e:
            logger.error(f"Error screening job {job.id} ({job.title}): {str(e)}")
            # Do not update the job state, so it can be retried later
            return {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "error": str(e)
            }


async def run_fast_screening_batch(
    batch_size: int,
    repo: JobRepository,
    time_budget_seconds: Optional[float] = None,
    concurrency: Optional[int] = None
) -> Dict[str, Any]:
    """Screen pending jobs against the user profile with native batching.

    Processes jobs in small waves (SCREENING_CHUNK_SIZE) under a strict semaphore.
    If the time budget is exceeded, returns clear partial results instead of
    blocking the caller indefinitely. Visa-friendly jobs are processed first.
    """
    # 1. Fetch user profile
    user_profile = await repo.get_user_profile()
    if not user_profile:
        raise ValueError("No user profile found. Please sync your CV first using the sync_cv tool.")
        
    profile = user_profile.profile
    
    # 2. Fetch prioritized pending jobs (visa-friendly first)
    jobs = await repo.get_jobs_by_state_prioritized("PENDING_SCREENING", limit=batch_size)
    if not jobs:
        return {
            "message": "No jobs pending screening.",
            "processed_count": 0,
            "screened_count": 0,
            "relevant_count": 0,
            "rejected_count": 0,
            "pending_remaining": 0,
            "partial": False,
            "errors": []
        }
    
    total_pending = (await repo.get_pipeline_counts()).get("PENDING_SCREENING", 0)
        
    # 3. Load LLM
    llm = await get_llm_for_stage("fast_screening", repo)
    structured_llm = llm.with_structured_output(FastScreeningResult)
    
    # 4. Strict concurrency control
    semaphore = asyncio.Semaphore(concurrency or SCREENING_CONCURRENCY)
    budget = time_budget_seconds if time_budget_seconds is not None else SCREENING_TIME_BUDGET_SECONDS
    started_at = time.monotonic()
    
    results: List[Dict[str, Any]] = []
    partial = False
    
    # 5. Process in native chunks, checking the time budget between waves
    for i in range(0, len(jobs), SCREENING_CHUNK_SIZE):
        chunk = jobs[i:i + SCREENING_CHUNK_SIZE]
        tasks = [
            screen_single_job(
                job=job,
                candidate_profile_summary=profile.summary,
                candidate_stack=profile.core_stack,
                candidate_skills=profile.skills,
                candidate_experience=profile.experience_years,
                candidate_seniority=profile.seniority_level,
                structured_llm=structured_llm,
                semaphore=semaphore
            )
            for job in chunk
        ]
        chunk_results = await asyncio.gather(*tasks)
        # Persist sequentially to avoid concurrent flushes on the shared session
        for r in chunk_results:
            if "error" not in r:
                db_job = await repo.get_job(r["id"])
                if db_job is None:
                    r["error"] = "Job disappeared from database."
                    continue
                db_job.state = "RELEVANT" if r["is_relevant"] else "REJECTED"
                db_job.screening_reason = r["reason"]
                await repo.save_job(db_job)
        results.extend(chunk_results)
        
        # Stop early if we still have jobs left and the time budget is exhausted
        if i + SCREENING_CHUNK_SIZE < len(jobs) and (time.monotonic() - started_at) >= budget:
            partial = True
            break
    
    processed = len(results)
    relevant_count = 0
    rejected_count = 0
    errors = []
    
    for r in results:
        if "error" in r:
            errors.append(f"Job '{r['title']}': {r['error']}")
        elif r.get("is_relevant"):
            relevant_count += 1
        else:
            rejected_count += 1
    
    return {
        "message": "Partial screening completed within the time budget; call again to continue." if partial else "Screening completed.",
        "processed_count": processed,
        "screened_count": processed - len(errors),
        "relevant_count": relevant_count,
        "rejected_count": rejected_count,
        "pending_remaining": max(0, total_pending - (processed - len(errors))),
        "partial": partial,
        "time_budget_seconds": budget,
        "errors": errors
    }
