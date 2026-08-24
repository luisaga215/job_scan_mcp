import pytest
from unittest.mock import patch

from job_scan_mcp.models import Job, UserProfile, ParsedProfile, ProfileJob, DeepEvaluationResult
from job_scan_mcp.services.cv_tailor import TailoredCV, TailoredJob, TailoredBullet
from job_scan_mcp.services.evaluation import (
    run_deep_evaluation_batch
)
from conftest import MockStructuredLLM

@pytest.mark.asyncio
async def test_run_deep_evaluation_no_profile(test_repo):
    """Test that deep evaluation throws an error if no user profile is synced."""
    with pytest.raises(ValueError, match="No user profile found"):
        await run_deep_evaluation_batch(batch_size=5, repo=test_repo)

@pytest.mark.asyncio
@patch("job_scan_mcp.services.evaluation.get_llm_for_stage")
async def test_run_deep_evaluation_success(mock_get_llm, test_repo):
    """Test full LangGraph deep evaluation batch process and state transitions."""
    # 1. Setup user profile in database
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python", "PostgreSQL", "AWS"],
        core_stack=["Python", "AWS"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv text", profile)

    # 2. Add a RELEVANT job to the database
    job = Job(
        id="relevant_job_1",
        title="Django Developer",
        company="Fast Growing Co",
        location="Remote",
        description="Looking for Python, AWS, and Django experience. Hybrid/Remote. 5 rounds of interviews.",
        job_url="http://fgc.com/1",
        state="RELEVANT"
    )
    await test_repo.save_job(job)

    # 3. Setup mock deep evaluation LLM structured result
    mock_metrics = DeepEvaluationResult(
        fit_score=90,
        interview_probability=80,
        core_stack_overlap=["Python", "AWS"],
        true_seniority_alignment="Matches candidate's Mid level experience perfectly.",
        red_flags=["5 rounds of interviews is a high round count", "uncompensated on-call mentions"],
        application_friction="High",
        pros=["Great tech stack matching", "Remote options"],
        cons=["Long interview process"]
    )
    mock_llm = MockStructuredLLM(mock_metrics)
    mock_get_llm.return_value = mock_llm

    # 4. Call deep evaluation batch
    result = await run_deep_evaluation_batch(batch_size=5, repo=test_repo)

    # 5. Assertions on results dict
    assert result["evaluated_count"] == 1
    assert result["average_fit_score"] == 90.0
    assert len(result["errors"]) == 0

    # 6. Verify job fields in DB
    db_job = await test_repo.get_job("relevant_job_1")
    assert db_job is not None
    assert db_job.state == "EVALUATED"
    assert db_job.fit_score == 90
    assert db_job.interview_probability == 80
    assert db_job.application_friction == "High"
    assert db_job.true_seniority_alignment == "Matches candidate's Mid level experience perfectly."
    assert "Python" in db_job.core_stack_overlap
    assert "AWS" in db_job.core_stack_overlap
    assert "uncompensated on-call mentions" in db_job.red_flags
    assert len(db_job.pros) == 2
    assert "Long interview process" in db_job.cons
    assert db_job.evaluated_at is not None


@pytest.mark.asyncio
@patch("job_scan_mcp.services.evaluation.get_llm_for_stage")
async def test_run_deep_evaluation_handles_extraction_error(mock_get_llm, test_repo):
    """Test that an LLM extraction failure keeps the job in RELEVANT state for retry."""
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python", "AWS"],
        core_stack=["Python", "AWS"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv text", profile)

    job = Job(
        id="relevant_job_err",
        title="Django Developer",
        company="Fast Growing Co",
        location="Remote",
        description="Python + AWS role",
        job_url="http://fgc.com/err",
        state="RELEVANT"
    )
    await test_repo.save_job(job)

    class FailingLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, *args, **kwargs):
            raise RuntimeError("model exploded")

    mock_get_llm.return_value = FailingLLM()

    result = await run_deep_evaluation_batch(batch_size=5, repo=test_repo)

    assert result["evaluated_count"] == 0
    assert len(result["errors"]) == 1
    assert "model exploded" in result["errors"][0]
    assert result["pending_remaining"] == 1

    # Job must remain RELEVANT so it can be retried
    db_job = await test_repo.get_job("relevant_job_err")
    assert db_job.state == "RELEVANT"


@pytest.mark.asyncio
@patch("job_scan_mcp.services.evaluation.get_llm_for_stage")
async def test_run_deep_evaluation_partial_time_budget(mock_get_llm, test_repo):
    """Test that deep evaluation returns partial results when the time budget is exhausted."""
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python", "AWS"],
        core_stack=["Python", "AWS"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv text", profile)

    # 6 RELEVANT jobs: more than one evaluation chunk (chunk size = 5)
    for i in range(6):
        job = Job(
            id=f"relevant_job_{i}",
            title=f"Django Developer {i}",
            company="Fast Growing Co",
            location="Remote",
            description="Python + AWS role",
            job_url=f"http://fgc.com/{i}",
            state="RELEVANT"
        )
        await test_repo.save_job(job)

    mock_metrics = DeepEvaluationResult(
        fit_score=80,
        interview_probability=70,
        core_stack_overlap=["Python", "AWS"],
        true_seniority_alignment="Matches.",
        red_flags=[],
        application_friction="Low",
        pros=["Good stack"],
        cons=[]
    )
    mock_get_llm.return_value = MockStructuredLLM(mock_metrics)

    result = await run_deep_evaluation_batch(batch_size=6, repo=test_repo, time_budget_seconds=0.0)

    assert result["partial"] is True
    assert result["evaluated_count"] == 5
    assert result["pending_remaining"] == 1


@pytest.mark.asyncio
@patch("job_scan_mcp.services.evaluation.get_llm_for_stage")
async def test_deep_evaluation_generates_and_persists_tailored_cv(mock_get_llm, test_repo):
    """Integration: a RELEVANT job evaluated with a structured profile gets a tailored_cv_json persisted."""
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python", "AWS"],
        core_stack=["Python", "AWS"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer",
        experience=[ProfileJob(title="Backend", company="Co", date="2024 - Present", location="Remote",
                               bullets=["Built Python services.", "Managed AWS infra."])],
    )
    await test_repo.save_user_profile("Raw cv text", profile)

    job = Job(
        id="cv_eval_job_1",
        title="Python Backend Engineer",
        company="Stripe",
        location="Remote",
        description="Python + AWS microservices",
        job_url="http://stripe.com/cv",
        state="RELEVANT"
    )
    await test_repo.save_job(job)

    mock_metrics = DeepEvaluationResult(
        fit_score=90,
        interview_probability=80,
        core_stack_overlap=["Python", "AWS"],
        true_seniority_alignment="Matches.",
        red_flags=[],
        application_friction="Low",
        pros=["Great fit"],
        cons=[]
    )
    tailored_cv = TailoredCV(
        name="John Doe",
        contact=["john@example.com"],
        summary="Python backend engineer.",
        experience=[TailoredJob(title="Backend", company="Co", date="2024 - Present", location="Remote", bullets=[
            TailoredBullet(text="Built high-throughput Python services on AWS.", modified=True, match_reason="JD requires AWS"),
        ])],
    )

    class SchemaAwareLLM:
        def with_structured_output(self, schema):
            self._schema = schema
            return self

        async def ainvoke(self, prompt, *args, **kwargs):
            if self._schema is DeepEvaluationResult:
                return mock_metrics
            return tailored_cv

    mock_get_llm.return_value = SchemaAwareLLM()

    result = await run_deep_evaluation_batch(batch_size=5, repo=test_repo)

    assert result["evaluated_count"] == 1
    assert result["tailored_cv_generated"] == 1

    db_job = await test_repo.get_job("cv_eval_job_1")
    assert db_job.state == "EVALUATED"
    assert db_job.tailored_cv_json is not None
    assert '"modified": true' in db_job.tailored_cv_json or "modified" in db_job.tailored_cv_json
