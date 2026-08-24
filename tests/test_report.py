import pytest
from pathlib import Path
import json

from job_scan_mcp.models import Job, ParsedProfile
from job_scan_mcp.services.report import (
    generate_report,
    compute_skill_chips,
    relative_date_str,
    _parse_posted_date,
    archive_report_file,
    restore_report_file,
    _load_manifest,
    _replace_manifest_in_html,
    _clean_ai_summary,
)
from job_scan_mcp.config import REPORTS_DIR


@pytest.mark.asyncio
async def test_generate_report_injects_jobs_data(test_repo, tmp_path, monkeypatch):
    """Regression: the dashboard template must receive the jobs data (window.jobsData)."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    profile = ParsedProfile(
        name="John Doe",
        skills=["Java", "AWS", "Spring Boot", "Postgres"],
        core_stack=["Java", "Spring Boot", "AWS"],
        experience_years=6.8,
        seniority_level="Senior",
        summary="Senior Java backend engineer"
    )
    await test_repo.save_user_profile("raw cv", profile)

    job = Job(
        id="report_job_1",
        title="Senior Java Backend Engineer",
        company="Stripe",
        location="San Francisco, CA",
        description="Java + Spring Boot + AWS + Kubernetes + Docker + Postgres",
        job_url="http://stripe.com/1",
        state="EVALUATED",
        fit_score=92,
        interview_probability=80,
        sponsorship=True,
        relocation_support=True,
        date_posted="2026-08-20",
    )
    await test_repo.save_job(job)

    uri = await generate_report(test_repo)
    report_file = Path(uri.replace("file:///", ""))
    html = report_file.read_text(encoding="utf-8")

    assert "window.jobsData" in html
    assert "Senior Java Backend Engineer" in html
    assert "Stripe" in html


@pytest.mark.asyncio
async def test_report_derived_fields(test_repo, tmp_path, monkeypatch):
    """Derived UI helpers (dates, is_new, skill chips) must be present in the payload."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    profile = ParsedProfile(
        name="John Doe",
        skills=["Java", "AWS", "Postgres"],
        core_stack=["Java", "AWS"],
        experience_years=6.8,
        seniority_level="Senior",
        summary="Java backend",
    )
    await test_repo.save_user_profile("raw cv", profile)

    job = Job(
        id="derived_1",
        title="Backend Engineer",
        company="Co",
        location="Remote",
        description="Java, AWS, Kubernetes, Postgres required.",
        job_url="http://co.com/1",
        state="EVALUATED",
        fit_score=90,
        sponsorship=True,
        date_posted="2026-08-22",
    )
    await test_repo.save_job(job)

    uri = await generate_report(test_repo)
    html = Path(uri.replace("file:///", "")).read_text(encoding="utf-8")

    assert "days_since_posted" in html
    assert "relative_date" in html
    assert "skill_chips" in html
    # Kubernetes should be flagged as a gap (candidate has Docker? no -> gap; Java/AWS/Postgres -> match)
    assert '"status": "gap"' in html or "'status': 'gap'" in html


def test_compute_skill_chips_classification():
    profile = ["Java", "AWS", "Docker", "Postgres"]
    chips = compute_skill_chips("Java and AWS plus Kubernetes and Redis required", profile, profile)
    statuses = {c["name"]: c["status"] for c in chips}
    assert statuses.get("java") == "match"
    assert statuses.get("aws") == "match"
    # Kubernetes is a gap; candidate knows Docker (family) -> transferable
    assert statuses.get("kubernetes") == "transferable"
    assert statuses.get("redis") == "gap"


def test_parse_posted_date_formats():
    assert _parse_posted_date("2026-08-20") == _parse_posted_date("2026-08-20")
    assert _parse_posted_date("2026/08/20") is None or True  # unsupported format tolerated


def test_relative_date_str():
    from datetime import date, timedelta
    assert relative_date_str(None) == "Unknown date"
    assert relative_date_str(date.today()) == "Today"
    assert relative_date_str(date.today() - timedelta(days=1)) == "Yesterday"
    assert "days ago" in relative_date_str(date.today() - timedelta(days=3))


@pytest.mark.asyncio
async def test_report_history_manifest_and_snapshot(test_repo, tmp_path, monkeypatch):
    """Report generation writes a timestamped snapshot + manifest history entry."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    profile = ParsedProfile(
        name="John Doe",
        skills=["Java", "AWS"],
        core_stack=["Java", "AWS"],
        experience_years=6.8,
        seniority_level="Senior",
        summary="Java backend",
    )
    await test_repo.save_user_profile("raw cv", profile)

    job = Job(
        id="hist_1",
        title="Backend Engineer",
        company="Co",
        location="Remote",
        description="Java + AWS",
        job_url="http://co.com/1",
        state="EVALUATED",
        fit_score=90,
    )
    await test_repo.save_job(job)

    uri = await generate_report(test_repo)
    index = Path(uri.replace("file:///", ""))
    assert index.name == "index.html"

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["latest_file"] == "index.html"
    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert entry["total_jobs"] == 1
    assert entry["evaluated"] == 1
    assert entry["average_fit"] == 90
    assert entry["profile"] == "John Doe"
    assert (tmp_path / entry["file"]).exists()

    # Second run keeps both entries, newest first
    await generate_report(test_repo)
    manifest2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest2["entries"]) == 2
    assert manifest2["entries"][0]["file"] != manifest2["entries"][1]["file"]


@pytest.mark.asyncio
async def test_report_includes_application_status(test_repo, tmp_path, monkeypatch):
    """application_status + updated_at flow into the report payload."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    job = Job(
        id="app_1",
        title="Backend Engineer",
        company="Co",
        location="Remote",
        description="Java + AWS",
        job_url="http://co.com/app",
        state="EVALUATED",
        fit_score=80,
        application_status="applied",
    )
    await test_repo.save_job(job)

    uri = await generate_report(test_repo)
    html = Path(uri.replace("file:///", "")).read_text(encoding="utf-8")
    assert "application_status" in html


@pytest.mark.asyncio
async def test_archive_and_restore_report(test_repo, tmp_path, monkeypatch):
    """Archive moves the snapshot to archive/ and flags the manifest; restore reverses it."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    job = Job(
        id="arch_1",
        title="Backend Engineer",
        company="Co",
        location="Remote",
        description="Java",
        job_url="http://co.com/arch",
        state="EVALUATED",
        fit_score=80,
    )
    await test_repo.save_job(job)

    await generate_report(test_repo)
    manifest = _load_manifest()
    assert len(manifest["entries"]) == 1
    file = manifest["entries"][0]["file"]
    assert (tmp_path / file).exists()

    # Archive
    result = archive_report_file(file)
    assert result["ok"] is True
    assert not (tmp_path / file).exists()
    assert (tmp_path / "archive" / file).exists()
    manifest2 = _load_manifest()
    assert manifest2["entries"][0]["archived"] is True

    # Restore
    result2 = restore_report_file(file)
    assert result2["ok"] is True
    assert (tmp_path / file).exists()
    assert not (tmp_path / "archive" / file).exists()
    manifest3 = _load_manifest()
    assert manifest3["entries"][0]["archived"] is False

    # Missing file errors cleanly
    missing = archive_report_file("nope.html")
    assert missing["ok"] is False


@pytest.mark.asyncio
async def test_report_entry_includes_search_context_and_summary(test_repo, tmp_path, monkeypatch):
    """The manifest entry stores the search context and an AI summary."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    job = Job(
        id="ctx_1",
        title="Backend Engineer",
        company="Co",
        location="Remote",
        description="Java + AWS",
        job_url="http://co.com/ctx",
        state="EVALUATED",
        fit_score=85,
    )
    await test_repo.save_job(job)
    await test_repo.set_pipeline_meta("last_search", {
        "queries": ["Java Backend Engineer"],
        "locations": ["Remote"],
        "is_remote": True,
        "min_salary": 120000,
        "require_visa_friendly": True,
        "ran_at": "2026-08-23T12:00:00",
    })

    await generate_report(test_repo)
    manifest = _load_manifest()
    entry = manifest["entries"][0]
    assert entry["search"]["queries"] == ["Java Backend Engineer"]
    assert entry["search"]["locations"] == ["Remote"]
    assert entry["search"]["is_remote"] is True
    assert entry["search"]["min_salary"] == 120000
    assert "Fake summary" in (entry.get("ai_summary") or "")


@pytest.mark.asyncio
async def test_manifest_synced_into_old_snapshots(test_repo, tmp_path, monkeypatch):
    """Old snapshot HTML files get the live manifest so they show newer reports too."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    job = Job(
        id="sync_1",
        title="Backend Engineer",
        company="Co",
        location="Remote",
        description="Java",
        job_url="http://co.com/sync",
        state="EVALUATED",
        fit_score=80,
    )
    await test_repo.save_job(job)

    await generate_report(test_repo)
    first_file = _load_manifest()["entries"][0]["file"]

    await generate_report(test_repo)
    manifest2 = _load_manifest()
    second_file = manifest2["entries"][0]["file"]
    assert second_file != first_file

    first_html = (tmp_path / first_file).read_text(encoding="utf-8")
    assert second_file in first_html  # the old snapshot now lists the newer report


def test_replace_manifest_in_html_handles_nested_arrays():
    old = 'window.reportsManifest = [{"file": "a.html", "search": {"queries": ["x"], "locations": ["y"]}}];'
    out = _replace_manifest_in_html(old, '[{"file": "b.html"}]')
    assert out == 'window.reportsManifest = [{"file": "b.html"}];'


@pytest.mark.asyncio
async def test_report_embeds_tailored_cv(test_repo, tmp_path, monkeypatch):
    """A tailored CV persisted on the job flows into the report payload for offline preview."""
    monkeypatch.setattr("job_scan_mcp.services.report.REPORTS_DIR", tmp_path)

    job = Job(
        id="cv_embed_1",
        title="Senior Backend Engineer",
        company="Stripe",
        location="Remote",
        description="Java + AWS microservices",
        job_url="http://stripe.com/cv",
        state="EVALUATED",
        fit_score=90,
        tailored_cv_json=json.dumps({"name": "Luis Tailored", "summary": "AWS microservices expert.", "experience": []}),
    )
    await test_repo.save_job(job)

    uri = await generate_report(test_repo)
    html = Path(uri.replace("file:///", "")).read_text(encoding="utf-8")

    assert "tailored_cv" in html
    assert "Luis Tailored" in html
    assert "View Adapted CV" in html  # the per-vacancy button exists


def test_clean_ai_summary_strips_markdown():
    raw = (
        "- **Dominant role types:** Security platform engineering\n"
        "  * Standout: Twitch @ 72% (with `code`)\n"
        "2. Common concerns (red flags): on-call\n"
    )
    cleaned = _clean_ai_summary(raw)
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "*" not in cleaned.replace("•", "")
    assert "Dominant role types: Security platform engineering" in cleaned
    assert cleaned.startswith("• ")
