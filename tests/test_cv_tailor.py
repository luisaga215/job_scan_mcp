import json
import pytest
from pathlib import Path
from unittest.mock import patch

from job_scan_mcp.services import cv_tailor
from job_scan_mcp.services.cv_tailor import (
    TailoredCV,
    TailoredJob,
    TailoredBullet,
    build_tailored_cv_data,
    build_base_cv_from_profile,
    export_cv_to_pdf,
    render_preview_html,
    _safe_filename,
    _strip_placeholders,
)
from job_scan_mcp.models import ParsedProfile, ProfileJob

SAMPLE_JD = (
    "Senior Java Backend Engineer at Stripe. Requires 6+ years building high-throughput "
    "microservices with Java and Spring Boot on AWS (SQS, SNS, RDS). Experience with "
    "distributed systems, canary deployments and cross-region replication."
)


def _base_cv():
    return {
        "name": "Luis Angel Gonzalez",
        "contact": ["luis.aga215@gmail.com", "Mexico City"],
        "summary": "Backend Engineer",
        "experience": [
            {
                "title": "System Development Engineer II",
                "company": "Amazon (Audible)",
                "date": "Sep 2025 - Present",
                "location": "Mexico City",
                "bullets": [
                    "Designed backend services for a canary deployment stage.",
                    "Orchestrated cross-region replication of services.",
                    "Implemented auth-token security across services.",
                ],
            }
        ],
        "education": [{"degree": "M.S. AI", "institution": "UNIR", "year": "2022"}],
        "skills": {"Technical": ["Java", "AWS", "Spring Boot"]},
    }


def _mock_structured_llm(tailored: TailoredCV):
    class _Mock:
        def __init__(self):
            self.last_prompt = None

        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt):
            self.last_prompt = prompt
            return tailored

    return _Mock()


def _mock_playwright():
    """Fake async playwright that records pdf() calls and creates the output file."""
    class FakePage:
        def __init__(self):
            self.pdf_calls = []

        async def goto(self, url):
            self.url = url

        async def pdf(self, **kwargs):
            self.pdf_calls.append(kwargs)
            Path(kwargs["path"]).write_bytes(b"%PDF-fake")

    class FakeBrowser:
        def __init__(self):
            self.page = FakePage()

        async def new_page(self):
            return self.page

        async def close(self):
            pass

    class FakeChromium:
        def __init__(self):
            self.browser = FakeBrowser()

        async def launch(self):
            return self.browser

    class FakePlaywright:
        def __init__(self):
            self.chromium = FakeChromium()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    return FakePlaywright()


# =====================================================================
# Tool A: generate_tailored_cv
# =====================================================================
@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.get_llm_for_stage")
async def test_generate_tailored_cv_injects_modified_flags(mock_get_llm, test_repo):
    tailored = TailoredCV(
        name="Luis Angel Gonzalez",
        contact=["luis.aga215@gmail.com"],
        summary="Senior Java backend engineer focused on AWS and microservices.",
        experience=[
            TailoredJob(
                title="System Development Engineer II",
                company="Amazon (Audible)",
                date="Sep 2025 - Present",
                location="Mexico City",
                bullets=[
                    TailoredBullet(
                        text="Architected AWS microservices using SQS/SNS, cutting deployment blast radius by 95%.",
                        modified=True,
                        match_reason="JD requires microservices and AWS",
                    ),
                    TailoredBullet(text="Migrated 69+ packages from JDK 8 to JDK 17.", modified=False),
                ],
            )
        ],
        education=[],
        skills={"Technical": ["Java", "AWS"]},
    )
    mock_get_llm.return_value = _mock_structured_llm(tailored)

    data = await build_tailored_cv_data(SAMPLE_JD, _base_cv(), test_repo)

    exp = data["experience"][0]
    modified = [b for b in exp["bullets"] if b["modified"]]
    unchanged = [b for b in exp["bullets"] if not b["modified"]]

    assert len(modified) == 1
    assert modified[0]["match_reason"] == "JD requires microservices and AWS"
    assert unchanged[0]["match_reason"] is None
    assert data["_meta"]["modified_bullets"] == 1
    assert data["_meta"]["total_bullets"] == 2


@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.get_llm_for_stage")
async def test_build_tailored_cv_data_requires_inputs(mock_get_llm, test_repo):
    # Empty JD is tolerated: returns the base CV unchanged (no fabricated changes)
    data = await build_tailored_cv_data("", _base_cv(), test_repo)
    assert data["_meta"]["modified_bullets"] == 0
    assert len(data["experience"]) == len(_base_cv()["experience"])

    # Empty base CV is still an error
    with pytest.raises(ValueError, match="base_cv_json"):
        await build_tailored_cv_data(SAMPLE_JD, {}, test_repo)


@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.get_llm_for_stage")
async def test_generate_tailored_cv_prompt_enforces_recruiter_standards(mock_get_llm, test_repo):
    tailored = TailoredCV(name="Luis", contact=["a@b.c"], summary="s")
    mock_llm = _mock_structured_llm(tailored)
    mock_get_llm.return_value = mock_llm

    await build_tailored_cv_data(SAMPLE_JD, _base_cv(), test_repo)
    prompt = mock_llm.last_prompt

    for required in [
        "TN Visa Eligible",
        "Open to US Relocation",
        "Commissioning Engineer",
        "Mon YYYY - Present",
        "XYZ",
        "NEVER use placeholders",
        "[X]%",
        "Languages & Frameworks",
        "Cloud & Infrastructure",
        "Architecture",
        "PRESERVE EVERY role",
    ]:
        assert required in prompt, f"prompt must require '{required}'"


def test_strip_placeholders_removes_metric_placeholders():
    assert _strip_placeholders("Cut latency by [X]% across services.") == "Cut latency by across services."
    assert _strip_placeholders("Managing [X]+ TPS for canary deployments.") == "Managing for canary deployments."
    assert _strip_placeholders("Reduced [Métrica] by [X].") == "Reduced by."
    assert _strip_placeholders("Plain bullet, no placeholders.") == "Plain bullet, no placeholders."


@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.get_llm_for_stage")
async def test_build_tailored_cv_data_strips_placeholders_from_output(mock_get_llm, test_repo):
    tailored = TailoredCV(
        name="Luis",
        contact=["a@b.c"],
        summary="s",
        experience=[TailoredJob(title="SysDev II", company="Amazon", bullets=[
            TailoredBullet(text="Cut latency by [X]% and managed [X]+ TPS.", modified=True, match_reason="JD wants metrics"),
        ])],
    )
    mock_get_llm.return_value = _mock_structured_llm(tailored)
    data = await build_tailored_cv_data(SAMPLE_JD, _base_cv(), test_repo)
    bullet = data["experience"][0]["bullets"][0]
    assert "[X]" not in bullet["text"]
    assert "TPS" not in bullet["text"]


# =====================================================================
# Tool B: export_cv_to_pdf
# =====================================================================
@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.async_playwright")
async def test_export_cv_to_pdf_uses_playwright(mock_pw, test_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(cv_tailor.config, "DATA_DIR", tmp_path)
    fake = _mock_playwright()
    mock_pw.return_value = fake

    path = await export_cv_to_pdf({"name": "Luis", "experience": []}, "My-Tailored CV")

    assert path.endswith("My-Tailored_CV.pdf")
    assert Path(path).exists()
    browser = await fake.chromium.launch()
    assert len(browser.page.pdf_calls) == 1
    assert browser.page.pdf_calls[0]["path"] == path
    # temp html cleaned up
    assert not list((tmp_path / "cv_tailor").glob("_tmp_*.html"))


def test_safe_filename():
    assert _safe_filename("My-Tailored CV") == "My-Tailored_CV"
    assert _safe_filename("cv.pdf") == "cv"
    assert _safe_filename("") == "tailored-cv"


def test_build_base_cv_from_profile():
    profile = ParsedProfile(
        name="Luis Angel Gonzalez",
        email="luis@example.com",
        skills=["Java", "AWS"],
        core_stack=["Java", "Spring Boot"],
        summary="Backend engineer",
        education=["M.S. AI"],
        experience=[
            ProfileJob(title="SysDev II", company="Amazon", date="Sep 2025 - Present",
                       location="Mexico City", bullets=["Designed canary deployments.", "Cross-region replication."]),
        ],
    )
    base = build_base_cv_from_profile(profile)
    assert base["name"] == "Luis Angel Gonzalez"
    assert "luis@example.com" in base["contact"]
    assert base["experience"][0]["company"] == "Amazon"
    assert len(base["experience"][0]["bullets"]) == 2
    assert base["skills"]["Technical"] == ["Java", "AWS"]


def test_build_base_cv_from_profile_none():
    assert build_base_cv_from_profile(None) == {}


def test_pdf_template_renders_bullet_dicts_and_strings():
    from jinja2 import Template
    html = Template(cv_tailor.PDF_TEMPLATE).render(cv={
        "name": "Luis",
        "contact": ["a@b.c"],
        "summary": "s",
        "experience": [
            {"title": "T", "company": "C", "date": "D", "location": "L",
             "bullets": [{"text": "modified bullet", "modified": True}, "plain bullet"]}
        ],
    })
    assert "modified bullet" in html
    assert "plain bullet" in html
    assert 'highlight' not in html  # PDF is always clean


# =====================================================================
# Preview rendering
# =====================================================================
@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.get_llm_for_stage")
async def test_render_preview_embeds_modified_flags(mock_get_llm, test_repo, tmp_path):
    tailored = TailoredCV(
        name="Luis", contact=["a@b.c"], summary="s",
        experience=[TailoredJob(title="T", company="C", bullets=[
            TailoredBullet(text="new", modified=True, match_reason="JD needs X"),
            TailoredBullet(text="same", modified=False),
        ])],
    )
    mock_get_llm.return_value = _mock_structured_llm(tailored)
    data = await build_tailored_cv_data(SAMPLE_JD, _base_cv(), test_repo)

    preview = render_preview_html(data, preview_path=tmp_path / "preview.html")
    html = preview.read_text(encoding="utf-8")
    assert '"modified": true' in html or '"modified":true' in html.replace(" ", "")
    assert "match_reason" in html
    assert "bulletCls" in html


# =====================================================================
# Integration: generate -> preview -> export
# =====================================================================
@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_tailor.async_playwright")
@patch("job_scan_mcp.services.cv_tailor.get_llm_for_stage")
async def test_full_flow_generate_preview_export(mock_get_llm, mock_pw, test_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(cv_tailor.config, "DATA_DIR", tmp_path)

    tailored = TailoredCV(
        name="Luis", contact=["a@b.c"], summary="Senior Java backend engineer.",
        experience=[TailoredJob(title="SysDev II", company="Amazon", bullets=[
            TailoredBullet(text="AWS microservices, SQS/SNS.", modified=True, match_reason="JD requires AWS"),
        ])],
    )
    mock_get_llm.return_value = _mock_structured_llm(tailored)
    mock_pw.return_value = _mock_playwright()

    data = await build_tailored_cv_data(SAMPLE_JD, _base_cv(), test_repo)
    preview = render_preview_html(data, preview_path=tmp_path / "cv_preview.html")
    assert preview.exists()

    pdf_path = await export_cv_to_pdf(data, "tailored-final")
    assert Path(pdf_path).exists()
    assert Path(pdf_path).suffix == ".pdf"