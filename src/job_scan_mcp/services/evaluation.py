import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from job_scan_mcp.models import Job, DeepEvaluationResult
from job_scan_mcp.repository import JobRepository
from job_scan_mcp.services.llm_factory import get_llm_for_stage

logger = logging.getLogger(__name__)

# =====================================================================
# Strict concurrency & resilience controls
# =====================================================================
EVALUATION_CONCURRENCY = 3            # Max concurrent LangGraph evaluations (rate-limit safe)
EVALUATION_JOB_TIMEOUT_SECONDS = 60   # Hard per-job timeout (prevents hangs blocking the batch)
EVALUATION_TIME_BUDGET_SECONDS = 25   # Return partial results after this budget, so the MCP client never times out
EVALUATION_CHUNK_SIZE = 5             # Jobs per scheduling wave

# =====================================================================
# LangGraph State & Node Definitions
# =====================================================================

class EvaluationState(TypedDict):
    job_id: str
    job_title: str
    job_company: str
    job_description: str
    user_profile: Dict[str, Any]
    llm: Any
    evaluation_result: Optional[Dict[str, Any]]
    errors: List[str]

async def extract_metrics_node(state: EvaluationState) -> Dict[str, Any]:
    """Node that uses the deep evaluation LLM to extract fit metrics."""
    job_desc = state["job_description"]
    profile = state["user_profile"]
    llm = state["llm"]
    
    # Enable structured output using the DeepEvaluationResult Pydantic schema
    try:
        structured_llm = llm.with_structured_output(DeepEvaluationResult)
    except Exception as e:
        logger.warning(f"with_structured_output failed, falling back to manual parsing or direct call: {str(e)}")
        structured_llm = llm  # Fallback
        
    prompt = (
        "You are an elite software engineering hiring manager and career strategist. "
        "Your task is to conduct a highly critical, realistic evaluation of a job posting against a candidate's profile.\n\n"
        "=== CANDIDATE PROFILE ===\n"
        f"Name: {profile.get('name')}\n"
        f"Seniority Level: {profile.get('seniority_level')}\n"
        f"Years of Experience: {profile.get('experience_years')}\n"
        f"Core Stack: {', '.join(profile.get('core_stack', []))}\n"
        f"Skills: {', '.join(profile.get('skills', []))}\n"
        f"Summary: {profile.get('summary')}\n\n"
        "=== JOB POSTING ===\n"
        f"Title: {state['job_title']}\n"
        f"Company: {state['job_company']}\n"
        f"Description:\n{job_desc}\n\n"
        "Analyze the job posting critically and output a structured DeepEvaluationResult:\n"
        "1. fit_score: Overall percentage match (0-100). Be realistic; do not inflate scores.\n"
        "2. interview_probability: Likelihood (0-100) of getting an interview based on stack and experience.\n"
        "3. core_stack_overlap: Specific languages, databases, cloud services, and tools from the job listing that match the candidate's core stack.\n"
        "4. true_seniority_alignment: A brief sentence identifying if this role aligns with the user's level (e.g., if a Mid-level role demands Principal tasks, or senior title is entry-level work).\n"
        "5. red_flags: List potential issues (e.g. uncompensated on-call, high turnover, 'fast paced', 'wear many hats').\n"
        "6. application_friction: Low (easy apply), Medium (simple application form), or High (requires take-home tasks, multiple essay questions, or 5+ interview rounds).\n"
        "7. pros: Key benefits of the role relative to candidate profile.\n"
        "8. cons: Key challenges or drawbacks of the role relative to candidate profile."
    )
    
    try:
        if hasattr(structured_llm, "ainvoke"):
            result = await structured_llm.ainvoke(prompt)
        else:
            result = await llm.ainvoke(prompt)
            
        if isinstance(result, DeepEvaluationResult):
            return {"evaluation_result": result.model_dump(), "errors": []}
        elif isinstance(result, dict):
            return {"evaluation_result": result, "errors": []}
        else:
            # Try parsing output if raw string returned
            import json
            raw_text = str(result.content) if hasattr(result, "content") else str(result)
            # Find json blocks if any
            if "{" in raw_text:
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                parsed = json.loads(raw_text[start:end])
                return {"evaluation_result": parsed, "errors": []}
            raise ValueError(f"Could not parse structured output from LLM result: {result}")
    except Exception as e:
        logger.error(f"LLM metrics extraction failed for job {state['job_id']}: {str(e)}")
        return {"errors": [f"LLM extraction failed: {str(e)}"]}

async def verify_metrics_node(state: EvaluationState) -> Dict[str, Any]:
    """Node that sanitizes, validates, and cleans extracted metrics."""
    if state.get("errors"):
        return {}
        
    result = state.get("evaluation_result")
    if not result:
        return {"errors": ["No evaluation result found to verify."]}
        
    # Standardize data values
    try:
        fit = int(result.get("fit_score", 0))
        result["fit_score"] = max(0, min(100, fit))
    except (ValueError, TypeError):
        result["fit_score"] = 50
        
    try:
        prob = int(result.get("interview_probability", 0))
        result["interview_probability"] = max(0, min(100, prob))
    except (ValueError, TypeError):
        result["interview_probability"] = 30
        
    friction = str(result.get("application_friction", "Medium")).strip().capitalize()
    if friction not in ["Low", "Medium", "High"]:
        result["application_friction"] = "Medium"
    else:
        result["application_friction"] = friction
        
    # Ensure lists are clean list structures
    for field in ["core_stack_overlap", "red_flags", "pros", "cons"]:
        val = result.get(field)
        if not isinstance(val, list):
            result[field] = [str(val)] if val else []
            
    return {"evaluation_result": result}

# Compile the LangGraph
workflow = StateGraph(EvaluationState)
workflow.add_node("extract_metrics", extract_metrics_node)
workflow.add_node("verify_metrics", verify_metrics_node)

workflow.set_entry_point("extract_metrics")
workflow.add_edge("extract_metrics", "verify_metrics")
workflow.add_edge("verify_metrics", END)

evaluation_graph = workflow.compile()


# =====================================================================
# Evaluation Service Implementation
# =====================================================================

async def evaluate_single_job(
    job: Job,
    profile_dict: Dict[str, Any],
    llm: Any,
    semaphore: asyncio.Semaphore,
    job_timeout: float = EVALUATION_JOB_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Execute LangGraph evaluation for a single job with rate-limit and timeout guarding.

    Runs the graph only; DB persistence happens sequentially in the caller to avoid
    concurrent flushes on a shared SQLAlchemy session.
    """
    async with semaphore:
        initial_state: EvaluationState = {
            "job_id": job.id,
            "job_title": job.title,
            "job_company": job.company,
            "job_description": job.description,
            "user_profile": profile_dict,
            "llm": llm,
            "evaluation_result": None,
            "errors": []
        }
        
        try:
            final_state = await asyncio.wait_for(
                evaluation_graph.ainvoke(initial_state),
                timeout=job_timeout
            )
            
            if final_state.get("errors"):
                return {
                    "id": job.id,
                    "title": job.title,
                    "company": job.company,
                    "errors": final_state["errors"]
                }
                
            metrics = final_state["evaluation_result"]
            return {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "fit_score": metrics.get("fit_score"),
                "interview_probability": metrics.get("interview_probability"),
                "metrics": metrics
            }
        except asyncio.TimeoutError:
            logger.error(f"Evaluation job {job.id} ({job.title}) timed out after {job_timeout}s")
            return {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "errors": [f"LangGraph evaluation timed out after {job_timeout}s"]
            }
        except Exception as e:
            logger.error(f"Failed to evaluate job {job.id}: {str(e)}")
            return {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "errors": [str(e)]
            }

async def run_deep_evaluation_batch(
    batch_size: int,
    repo: JobRepository,
    time_budget_seconds: Optional[float] = None,
    concurrency: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate RELEVANT jobs through LangGraph with native batching.

    Processes jobs in small waves (EVALUATION_CHUNK_SIZE) under a strict semaphore.
    If the time budget is exceeded, returns clear partial results instead of
    blocking the caller indefinitely.
    """
    # 1. Fetch user profile
    user_profile = await repo.get_user_profile()
    if not user_profile:
        raise ValueError("No user profile found. Please sync your CV first using the sync_cv tool.")
        
    profile_dict = user_profile.profile.model_dump()
    
    # 2. Fetch relevant jobs
    jobs = await repo.get_jobs_by_state("RELEVANT", limit=batch_size)
    if not jobs:
        return {
            "message": "No relevant jobs pending deep evaluation.",
            "evaluated_count": 0,
            "average_fit_score": 0.0,
            "pending_remaining": 0,
            "partial": False,
            "errors": []
        }
        
    total_relevant = (await repo.get_pipeline_counts()).get("RELEVANT", 0)
    
    # 3. Fetch deep evaluation LLM
    llm = await get_llm_for_stage("deep_evaluation", repo)
    
    # 4. Strict concurrency control
    semaphore = asyncio.Semaphore(concurrency or EVALUATION_CONCURRENCY)
    budget = time_budget_seconds if time_budget_seconds is not None else EVALUATION_TIME_BUDGET_SECONDS
    started_at = time.monotonic()
    
    results: List[Dict[str, Any]] = []
    partial = False
    
    # 5. Process in native chunks, checking the time budget between waves
    for i in range(0, len(jobs), EVALUATION_CHUNK_SIZE):
        chunk = jobs[i:i + EVALUATION_CHUNK_SIZE]
        tasks = [
            evaluate_single_job(
                job=job,
                profile_dict=profile_dict,
                llm=llm,
                semaphore=semaphore
            )
            for job in chunk
        ]
        chunk_results = await asyncio.gather(*tasks)
        # Persist sequentially to avoid concurrent flushes on the shared session
        for r in chunk_results:
            if not r.get("errors") and r.get("metrics"):
                db_job = await repo.get_job(r["id"])
                if db_job is None:
                    r["errors"] = ["Job disappeared from database."]
                    continue
                metrics = r["metrics"]
                db_job.state = "EVALUATED"
                db_job.fit_score = metrics.get("fit_score")
                db_job.interview_probability = metrics.get("interview_probability")
                db_job.core_stack_overlap = metrics.get("core_stack_overlap", [])
                db_job.true_seniority_alignment = metrics.get("true_seniority_alignment")
                db_job.red_flags = metrics.get("red_flags", [])
                db_job.application_friction = metrics.get("application_friction")
                db_job.pros = metrics.get("pros", [])
                db_job.cons = metrics.get("cons", [])
                db_job.evaluated_at = datetime.utcnow()
                await repo.save_job(db_job)
        results.extend(chunk_results)
        
        # Stop early if we still have jobs left and the time budget is exhausted
        if i + EVALUATION_CHUNK_SIZE < len(jobs) and (time.monotonic() - started_at) >= budget:
            partial = True
            break
    
    evaluated_count = 0
    total_fit_score = 0
    errors = []
    
    for r in results:
        if r.get("errors"):
            errors.append(f"Job '{r['title']}' ({r['company']}): {', '.join(r['errors'])}")
        else:
            evaluated_count += 1
            total_fit_score += r.get("fit_score", 0)
            
    avg_fit = (total_fit_score / evaluated_count) if evaluated_count > 0 else 0.0
    
    return {
        "message": "Partial deep evaluation completed within the time budget; call again to continue." if partial else "Deep evaluation completed.",
        "evaluated_count": evaluated_count,
        "average_fit_score": round(avg_fit, 2),
        "pending_remaining": max(0, total_relevant - evaluated_count),
        "partial": partial,
        "time_budget_seconds": budget,
        "errors": errors
    }
