import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from job_scan_mcp.database import db_manager
from job_scan_mcp.models import ParsedProfile, DeepEvaluationResult
from job_scan_mcp.mcp_server import (
    configure_llm,
    sync_cv,
    get_user_profile,
    fetch_and_filter_jobs,
    run_fast_screening,
    run_deep_evaluation,
    get_pipeline_status,
    set_job_application_status,
    generate_html_report
)
from conftest import MockStructuredLLM

@pytest.fixture(scope="function", autouse=True)
async def setup_integration_db():
    """Auto-initialize the overridden global in-memory database before each integration test."""
    await db_manager.init_db()
    yield
    # No close necessary, engine is cleaned up when test process ends

@pytest.mark.asyncio
async def test_configure_llm_tool():
    """Test configure_llm tool validation and persistence."""
    # Test valid configuration
    res = await configure_llm(stage="fast_screening", provider="openai", model="gpt-4o-mini", base_url="http://custom.url")
    assert res["status"] == "success"
    assert res["config"]["provider"] == "openai"
    assert res["config"]["model"] == "gpt-4o-mini"
    assert res["config"]["base_url"] == "http://custom.url"

    # Test invalid stage
    res_err = await configure_llm(stage="invalid_stage", provider="openai", model="gpt-4o")
    assert res_err["status"] == "error"
    assert "stage" in res_err["message"].lower()

@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_service.extract_text_from_pdf")
@patch("job_scan_mcp.services.cv_service.get_llm_for_stage")
async def test_sync_cv_and_profile_tools(mock_get_llm, mock_extract_pdf):
    """Test sync_cv tool parsing and profile retrieval."""
    # 1. Setup mock CV extraction
    mock_extract_pdf.return_value = "Jane Doe resume details"
    
    # 2. Setup mock LLM structured result
    parsed = ParsedProfile(
        name="Jane Doe",
        email="jane@example.com",
        skills=["Python", "SQL"],
        core_stack=["Python"],
        experience_years=5.0,
        seniority_level="Senior",
        summary="Senior python dev"
    )
    mock_get_llm.return_value = MockStructuredLLM(parsed)

    # 3. Call sync tool
    sync_res = await sync_cv(file_path="resume.pdf")
    assert sync_res["status"] == "success"
    assert sync_res["profile"]["name"] == "Jane Doe"

    # 4. Call get user profile tool
    profile_res = await get_user_profile()
    assert profile_res["status"] == "success"
    assert profile_res["profile"]["name"] == "Jane Doe"

@pytest.mark.asyncio
@patch("job_scan_mcp.services.job_service.scrape_jobs")
async def test_fetch_and_filter_jobs_tool(mock_scrape):
    """Test job scraping and filtering tools."""
    # Setup mock scraper dataframe
    mock_data = pd.DataFrame([
        {
            "title": "Backend Engineer",
            "company": "Stripe",
            "location": "Remote",
            "description": "Python Stripe API coding",
            "job_url": "https://stripe.com/jobs/1",
            "min_amount": 140000,
            "max_amount": 180000,
            "currency": "USD",
            "is_remote": True,
            "date_posted": "2026-08-22"
        }
    ])
    mock_scrape.return_value = mock_data

    # Call tool
    res = await fetch_and_filter_jobs(
        queries=["Backend Engineer"],
        locations=["Remote"],
        max_pool_size=10,
        is_remote=True,
        min_salary=100000
    )
    assert res["status"] == "success"
    assert res["data"]["total_scraped"] == 1
    assert res["data"]["new_jobs_saved"] == 1

@pytest.mark.asyncio
@patch("job_scan_mcp.services.screening.get_llm_for_stage")
@patch("job_scan_mcp.services.evaluation.get_llm_for_stage")
async def test_pipeline_execution_and_status(mock_get_llm_eval, mock_get_llm_screen):
    """Test full fast screening and deep evaluation tool pipeline with status checking."""
    # Setup mock fast screening result
    from job_scan_mcp.services.screening import FastScreeningResult
    mock_screen_res = FastScreeningResult(is_relevant=True, reason="Fits tech requirements")
    mock_get_llm_screen.return_value = MockStructuredLLM(mock_screen_res)

    # Setup mock deep evaluation result
    mock_eval_res = DeepEvaluationResult(
        fit_score=85,
        interview_probability=75,
        core_stack_overlap=["Python"],
        true_seniority_alignment="Matches candidate's Senior level experience",
        red_flags=["None"],
        application_friction="Low",
        pros=["Great compensation"],
        cons=["Long hours"]
    )
    mock_get_llm_eval.return_value = MockStructuredLLM(mock_eval_res)

    # First, run fast screening tool
    screen_res = await run_fast_screening(batch_size=5)
    assert screen_res["status"] == "success"
    # Even if DB is empty, should run and return screened_count = 0 or 1 depending on previous test
    
    # Run deep evaluation tool
    eval_res = await run_deep_evaluation(batch_size=5)
    assert eval_res["status"] == "success"

    # Fetch status report
    status_res = await get_pipeline_status()
    assert status_res["status"] == "success"
    assert "pipeline_counts" in status_res
    assert "llm_configurations" in status_res

@pytest.mark.asyncio
@patch("job_scan_mcp.services.screening.get_llm_for_stage")
async def test_run_fast_screening_tool_accepts_batching_params(mock_get_llm_screen):
    """Test that screening tool accepts time_budget_seconds and concurrency overrides."""
    from job_scan_mcp.services.screening import FastScreeningResult
    mock_screen_res = FastScreeningResult(is_relevant=True, reason="Fits tech requirements")
    mock_get_llm_screen.return_value = MockStructuredLLM(mock_screen_res)

    res = await run_fast_screening(batch_size=5, time_budget_seconds=30, concurrency=2)
    assert res["status"] == "success"
    assert "partial" in res["data"]

@pytest.mark.asyncio
@patch("job_scan_mcp.services.evaluation.get_llm_for_stage")
async def test_run_deep_evaluation_tool_accepts_batching_params(mock_get_llm_eval):
    """Test that deep evaluation tool accepts time_budget_seconds and concurrency overrides."""
    mock_eval_res = DeepEvaluationResult(
        fit_score=85,
        interview_probability=75,
        core_stack_overlap=["Python"],
        true_seniority_alignment="Matches.",
        red_flags=[],
        application_friction="Low",
        pros=["Great compensation"],
        cons=[]
    )
    mock_get_llm_eval.return_value = MockStructuredLLM(mock_eval_res)

    res = await run_deep_evaluation(batch_size=5, time_budget_seconds=30, concurrency=2)
    assert res["status"] == "success"
    assert "partial" in res["data"]

@pytest.mark.asyncio
async def test_set_job_application_status_tool():
    """Test the kanban persistence tool validates and updates job status."""
    from job_scan_mcp.repository import JobRepository
    from job_scan_mcp.database import db_manager
    from job_scan_mcp.models import Job

    async with db_manager.session() as session:
        repo = JobRepository(session)
        job = Job(
            id="kanban_1",
            title="Backend Engineer",
            company="Co",
            location="Remote",
            description="Java",
            job_url="http://co.com/k",
            state="EVALUATED",
            fit_score=90,
        )
        await repo.save_job(job)

    res = await set_job_application_status(job_id="kanban_1", status="interview")
    assert res["status"] == "success"
    assert res["job"]["application_status"] == "interview"

    bad = await set_job_application_status(job_id="kanban_1", status="nope")
    assert bad["status"] == "error"

    missing = await set_job_application_status(job_id="does_not_exist", status="applied")
    assert missing["status"] == "error"

    async with db_manager.session() as session:
        repo = JobRepository(session)
        updated = await repo.get_job("kanban_1")
        assert updated.application_status == "interview"
        assert updated.application_status_updated_at is not None

@pytest.mark.asyncio
async def test_generate_html_report_tool():
    """Test that the report compiles and returns a valid file URI path."""
    res = await generate_html_report()
    assert res["status"] == "success"
    assert res["report_path"].startswith("file://")
