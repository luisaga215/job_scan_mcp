import pytest
import pandas as pd
from unittest.mock import patch

from job_scan_mcp.services.job_service import (
    generate_job_id,
    apply_filters,
    analyze_visa_fit,
    fetch_and_save_jobs
)

def test_generate_job_id():
    """Verify deterministic hashing of company, title, and location."""
    hash1 = generate_job_id("Google", "Software Engineer", "Remote")
    hash2 = generate_job_id(" google ", "software engineer", "remote")
    hash3 = generate_job_id("Apple", "Software Engineer", "Remote")
    
    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA-256 hex string length

def test_apply_filters_remote():
    """Verify remote filtering heuristics."""
    # Remote job
    job_remote = {"is_remote": True, "location": "Austin, TX", "description": "some description"}
    job_heuristic_remote = {"is_remote": None, "location": "Remote", "description": "some description"}
    job_onsite = {"is_remote": False, "location": "Austin, TX", "description": "must work in office"}
    
    # 1. remote filter is True
    assert apply_filters(job_remote, is_remote_filter=True) is True
    assert apply_filters(job_heuristic_remote, is_remote_filter=True) is True
    assert apply_filters(job_onsite, is_remote_filter=True) is False
    
    # 2. remote filter is False
    assert apply_filters(job_remote, is_remote_filter=False) is False
    assert apply_filters(job_onsite, is_remote_filter=False) is True
    
    # 3. remote filter is None
    assert apply_filters(job_remote, is_remote_filter=None) is True
    assert apply_filters(job_onsite, is_remote_filter=None) is True

def test_apply_filters_salary():
    """Verify salary filtering with None-toleration."""
    job_no_salary = {"salary_min": None, "salary_max": None}
    job_low_salary = {"salary_min": 50000, "salary_max": 70000}
    job_high_salary = {"salary_min": 120000, "salary_max": 150000}
    
    # Tolerates None fields
    assert apply_filters(job_no_salary, min_salary_filter=100000) is True
    
    # Filters out if below
    assert apply_filters(job_low_salary, min_salary_filter=100000) is False
    
    # Keeps if above
    assert apply_filters(job_high_salary, min_salary_filter=100000) is True

def test_analyze_visa_fit_positive():
    """Detect explicit sponsorship and relocation signals."""
    job = {
        "location": "Austin, TX",
        "description": "We provide H-1B visa sponsorship and a relocation assistance package."
    }
    analysis = analyze_visa_fit(job)
    assert analysis["sponsorship"] is True
    assert analysis["relocation_support"] is True
    assert analysis["us_eligible"] is True
    assert "h-1b" in analysis["visa_keywords"]

def test_analyze_visa_fit_negative():
    """Detect explicit exclusions of candidates needing sponsorship."""
    job = {
        "location": "Seattle, WA",
        "description": "Must be authorized to work in the US. No visa sponsorship available."
    }
    analysis = analyze_visa_fit(job)
    assert analysis["sponsorship"] is False
    assert analysis["relocation_support"] is False
    assert analysis["us_eligible"] is False

def test_analyze_visa_fit_tn_visa():
    """TN visa mentions should count as sponsorship-friendly."""
    job = {
        "location": "New York, NY",
        "description": "Canadian and Mexican citizens eligible under TN visa."
    }
    analysis = analyze_visa_fit(job)
    assert analysis["sponsorship"] is True

def test_apply_filters_require_visa_friendly():
    """Strict mode keeps only postings with explicit sponsorship/relocation signals."""
    sponsored = {
        "is_remote": False,
        "location": "Austin, TX",
        "description": "Visa sponsorship offered.",
        "salary_min": 120000,
        "salary_max": 150000
    }
    hostile = {
        "is_remote": False,
        "location": "Seattle, WA",
        "description": "Must be authorized to work in the US.",
        "salary_min": 120000,
        "salary_max": 150000
    }
    unknown = {
        "is_remote": False,
        "location": "Boston, MA",
        "description": "Java backend role, no visa details.",
        "salary_min": 120000,
        "salary_max": 150000
    }
    assert apply_filters(sponsored, min_salary_filter=100000, require_visa_friendly=True) is True
    assert apply_filters(hostile, min_salary_filter=100000, require_visa_friendly=True) is False
    assert apply_filters(unknown, min_salary_filter=100000, require_visa_friendly=True) is False

@pytest.mark.asyncio
@patch("job_scan_mcp.services.job_service.scrape_jobs")
async def test_fetch_and_save_jobs(mock_scrape, test_repo):
    """Test full cycle: scrape, filter, and save to DB."""
    # 1. Setup mock DataFrame returned by jobspy
    mock_data = pd.DataFrame([
        {
            "title": "Software Engineer",
            "company": "Google",
            "location": "Remote",
            "description": "Python Backend developer",
            "job_url": "https://careers.google.com/jobs/1",
            "min_amount": 120000,
            "max_amount": 160000,
            "currency": "USD",
            "is_remote": True,
            "date_posted": "2026-08-22"
        },
        {
            "title": "Junior Python Dev",
            "company": "Startup Co",
            "location": "Boston, MA",
            "description": "Django scripting",
            "job_url": "https://startup.co/jobs/2",
            "min_amount": 50000,
            "max_amount": 60000,
            "currency": "USD",
            "is_remote": False,
            "date_posted": "2026-08-21"
        }
    ])
    mock_scrape.return_value = mock_data
    
    # 2. Call service with min_salary=100000 and is_remote=True
    # The Junior Dev should be filtered out because it is not remote and salary is too low.
    result = await fetch_and_save_jobs(
        repo=test_repo,
        queries=["Python Developer"],
        locations=["Remote"],
        max_pool_size=10,
        is_remote=True,
        min_salary=100000
    )
    
    # 3. Assertions on result stats
    assert result["total_scraped"] == 2
    assert result["total_filtered"] == 1
    assert result["new_jobs_saved"] == 1
    
    # 4. Assertions on database content
    jobs_in_db = await test_repo.get_jobs_by_state("PENDING_SCREENING")
    assert len(jobs_in_db) == 1
    saved_job = jobs_in_db[0]
    assert saved_job.company == "Google"
    assert saved_job.title == "Software Engineer"
    assert saved_job.salary_min == 120000.0


@pytest.mark.asyncio
@patch("job_scan_mcp.services.job_service.scrape_jobs")
async def test_fetch_and_save_jobs_saves_visa_flags(mock_scrape, test_repo):
    """Verify visa/relocation signals are computed and persisted on saved jobs."""
    mock_data = pd.DataFrame([
        {
            "title": "Backend Engineer",
            "company": "Sponsor Co",
            "location": "San Francisco, CA",
            "description": "H-1B visa sponsorship and relocation assistance available.",
            "job_url": "https://sponsor.co/jobs/1",
            "min_amount": 130000,
            "max_amount": 170000,
            "currency": "USD",
            "is_remote": False,
            "date_posted": "2026-08-22"
        },
        {
            "title": "Backend Engineer",
            "company": "Hostile Co",
            "location": "Seattle, WA",
            "description": "Must be authorized to work in the US. No sponsorship.",
            "job_url": "https://hostile.co/jobs/2",
            "min_amount": 140000,
            "max_amount": 180000,
            "currency": "USD",
            "is_remote": False,
            "date_posted": "2026-08-22"
        }
    ])
    mock_scrape.return_value = mock_data

    result = await fetch_and_save_jobs(
        repo=test_repo,
        queries=["Backend Engineer"],
        locations=["United States"],
        max_pool_size=10
    )

    assert result["new_jobs_saved"] == 2
    assert result["visa_friendly_saved"] == 1

    sponsor_job = await test_repo.get_job(generate_job_id("Sponsor Co", "Backend Engineer", "San Francisco, CA"))
    hostile_job = await test_repo.get_job(generate_job_id("Hostile Co", "Backend Engineer", "Seattle, WA"))

    assert sponsor_job.sponsorship is True
    assert sponsor_job.relocation_support is True
    assert "h-1b" in sponsor_job.visa_keywords

    assert hostile_job.sponsorship is False
    assert hostile_job.us_eligible is False


@pytest.mark.asyncio
@patch("job_scan_mcp.services.job_service.scrape_jobs")
async def test_fetch_and_save_jobs_strict_visa_mode(mock_scrape, test_repo):
    """Strict mode keeps only postings with explicit sponsorship/relocation signals."""
    mock_data = pd.DataFrame([
        {
            "title": "Backend Engineer",
            "company": "Sponsor Co",
            "location": "San Francisco, CA",
            "description": "H-1B visa sponsorship available.",
            "job_url": "https://sponsor.co/jobs/1",
            "min_amount": 130000,
            "max_amount": 170000,
            "currency": "USD",
            "is_remote": False,
            "date_posted": "2026-08-22"
        },
        {
            "title": "Backend Engineer",
            "company": "Plain Co",
            "location": "Boston, MA",
            "description": "Java backend role.",
            "job_url": "https://plain.co/jobs/2",
            "min_amount": 130000,
            "max_amount": 170000,
            "currency": "USD",
            "is_remote": False,
            "date_posted": "2026-08-22"
        }
    ])
    mock_scrape.return_value = mock_data

    result = await fetch_and_save_jobs(
        repo=test_repo,
        queries=["Backend Engineer"],
        locations=["United States"],
        max_pool_size=10,
        require_visa_friendly=True
    )

    assert result["total_filtered"] == 1
    assert result["new_jobs_saved"] == 1

    jobs_in_db = await test_repo.get_jobs_by_state("PENDING_SCREENING")
    assert len(jobs_in_db) == 1
    assert jobs_in_db[0].company == "Sponsor Co"
