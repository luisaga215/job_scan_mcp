import asyncio
import pytest
from unittest.mock import patch

from job_scan_mcp.models import Job, UserProfile, ParsedProfile
from job_scan_mcp.services.screening import (
    FastScreeningResult,
    run_fast_screening_batch
)
from conftest import MockStructuredLLM

@pytest.mark.asyncio
async def test_run_fast_screening_no_profile(test_repo):
    """Test that screening throws an error if no user profile is synced."""
    with pytest.raises(ValueError, match="No user profile found"):
        await run_fast_screening_batch(batch_size=5, repo=test_repo)

@pytest.mark.asyncio
@patch("job_scan_mcp.services.screening.get_llm_for_stage")
async def test_run_fast_screening_success(mock_get_llm, test_repo):
    """Test batch fast screening workflow and state transitions."""
    # 1. Setup user profile in database
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python"],
        core_stack=["Python"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv", profile)

    # 2. Add two pending jobs to the database
    job1 = Job(
        id="hash_job_1",
        title="Senior Python Dev",
        company="Tech Corp",
        location="Remote",
        description="Looking for 8 years of Python experience.",
        job_url="http://corp.com/1",
        state="PENDING_SCREENING"
    )
    job2 = Job(
        id="hash_job_2",
        title="Django Developer",
        company="App Inc",
        location="Remote",
        description="Looking for mid Python dev with Django experience.",
        job_url="http://inc.com/2",
        state="PENDING_SCREENING"
    )
    await test_repo.save_job(job1)
    await test_repo.save_job(job2)

    # 3. Setup mock LLM answers:
    # First job processed (Senior Python Dev) is rejected (too senior).
    # Second job (Django Developer) is relevant.
    # To mock successive calls of ainvoke, we can return dynamic values or create a simple mock.
    class SequentialMockLLM:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, *args, **kwargs):
            self.calls += 1
            if "Senior Python Dev" in prompt:
                return FastScreeningResult(is_relevant=False, reason="Requires more experience than candidate has.")
            else:
                return FastScreeningResult(is_relevant=True, reason="Excellent stack and mid-level seniority alignment.")

    mock_get_llm.return_value = SequentialMockLLM()

    # 4. Call service
    result = await run_fast_screening_batch(batch_size=10, repo=test_repo)

    # 5. Assertions
    assert result["screened_count"] == 2
    assert result["relevant_count"] == 1
    assert result["rejected_count"] == 1
    assert len(result["errors"]) == 0

    # 6. Verify DB states
    db_job1 = await test_repo.get_job("hash_job_1")
    db_job2 = await test_repo.get_job("hash_job_2")
    
    assert db_job1.state == "REJECTED"
    assert "more experience" in db_job1.screening_reason
    
    assert db_job2.state == "RELEVANT"
    assert "mid-level seniority" in db_job2.screening_reason


@pytest.mark.asyncio
@patch("job_scan_mcp.services.screening.get_llm_for_stage")
async def test_run_fast_screening_partial_time_budget(mock_get_llm, test_repo):
    """Test that screening returns partial results when the time budget is exhausted."""
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python"],
        core_stack=["Python"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv", profile)

    # Insert 15 pending jobs: more than one screening chunk (chunk size = 10)
    for i in range(15):
        job = Job(
            id=f"hash_job_{i}",
            title=f"Python Dev {i}",
            company="Tech Corp",
            location="Remote",
            description="Python backend role",
            job_url=f"http://corp.com/{i}",
            state="PENDING_SCREENING"
        )
        await test_repo.save_job(job)

    class InstantMockLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, *args, **kwargs):
            return FastScreeningResult(is_relevant=True, reason="Good stack fit.")

    mock_get_llm.return_value = InstantMockLLM()

    # A zero-second budget forces a partial return after the first chunk
    result = await run_fast_screening_batch(batch_size=15, repo=test_repo, time_budget_seconds=0.0)

    assert result["partial"] is True
    assert result["processed_count"] == 10
    assert result["pending_remaining"] == 5
    assert result["message"] != "Screening completed."


@pytest.mark.asyncio
@patch("job_scan_mcp.services.screening.get_llm_for_stage")
async def test_run_fast_screening_handles_timeout(mock_get_llm, test_repo):
    """Test that a hung LLM call is contained per-job and does not block the batch."""
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python"],
        core_stack=["Python"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv", profile)

    job1 = Job(
        id="hash_timeout_1",
        title="Python Dev",
        company="Corp A",
        location="Remote",
        description="Python role",
        job_url="http://a.com/1",
        state="PENDING_SCREENING"
    )
    job2 = Job(
        id="hash_timeout_2",
        title="Java Dev",
        company="Corp B",
        location="Remote",
        description="Java role",
        job_url="http://b.com/2",
        state="PENDING_SCREENING"
    )
    await test_repo.save_job(job1)
    await test_repo.save_job(job2)

    class HungLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, *args, **kwargs):
            raise asyncio.TimeoutError("simulated hang")

    mock_get_llm.return_value = HungLLM()

    result = await run_fast_screening_batch(batch_size=2, repo=test_repo, concurrency=1)

    assert result["screened_count"] == 0
    assert len(result["errors"]) == 2
    assert result["pending_remaining"] == 2

    # Jobs must remain PENDING_SCREENING so they can be retried
    assert (await test_repo.get_job("hash_timeout_1")).state == "PENDING_SCREENING"
    assert (await test_repo.get_job("hash_timeout_2")).state == "PENDING_SCREENING"


@pytest.mark.asyncio
@patch("job_scan_mcp.services.screening.get_llm_for_stage")
async def test_screening_prompt_includes_visa_context(mock_get_llm, test_repo):
    """Test that the screening prompt surfaces visa/sponsorship signals."""
    profile = ParsedProfile(
        name="John Doe",
        skills=["Python"],
        core_stack=["Python"],
        experience_years=5.0,
        seniority_level="Mid",
        summary="Mid Python Developer"
    )
    await test_repo.save_user_profile("Raw cv", profile)

    job = Job(
        id="hash_visa_1",
        title="Backend Engineer",
        company="Sponsor Co",
        location="San Francisco, CA",
        description="H-1B visa sponsorship available.",
        job_url="http://sponsor.com/1",
        state="PENDING_SCREENING",
        sponsorship=True,
        relocation_support=True,
        visa_keywords="h-1b, relocation assistance"
    )
    await test_repo.save_job(job)

    class RecordingLLM:
        def __init__(self):
            self.last_prompt = None

        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, *args, **kwargs):
            self.last_prompt = prompt
            return FastScreeningResult(is_relevant=True, reason="Fits.")

    llm = RecordingLLM()
    mock_get_llm.return_value = llm

    await run_fast_screening_batch(batch_size=1, repo=test_repo)

    assert "Visa/Sponsorship signals: h-1b, relocation assistance" in llm.last_prompt
    assert "Relocation support: yes" in llm.last_prompt


@pytest.mark.asyncio
async def test_repo_prioritizes_visa_friendly_jobs(test_repo):
    """Test that pending jobs are fetched with visa-friendly postings first."""
    job_a = Job(
        id="no_visa",
        title="Engineer A",
        company="Co A",
        location="Remote",
        description="Must be authorized to work in the US.",
        job_url="http://a.com",
        state="PENDING_SCREENING",
        sponsorship=False
    )
    job_b = Job(
        id="visa_yes",
        title="Engineer B",
        company="Co B",
        location="Austin, TX",
        description="We sponsor H-1B.",
        job_url="http://b.com",
        state="PENDING_SCREENING",
        sponsorship=True
    )
    job_c = Job(
        id="visa_unknown",
        title="Engineer C",
        company="Co C",
        location="Remote",
        description="Generic role",
        job_url="http://c.com",
        state="PENDING_SCREENING"
    )
    await test_repo.save_job(job_a)
    await test_repo.save_job(job_b)
    await test_repo.save_job(job_c)

    ordered = await test_repo.get_jobs_by_state_prioritized("PENDING_SCREENING", limit=10)
    ids = [j.id for j in ordered]

    assert ids[0] == "visa_yes"
