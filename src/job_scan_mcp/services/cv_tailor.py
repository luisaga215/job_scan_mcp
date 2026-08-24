"""Dynamic CV Tailoring Engine.

Two-stage architecture:
  - generate_tailored_cv: LLM inference that adapts a base CV to a job
    description, flagging every modified node with ``modified`` + ``match_reason``.
  - export_cv_to_pdf: renders the tailored JSON into an ATS-friendly HTML
    template (Playwright) and exports it to PDF.
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from jinja2 import Template
from playwright.async_api import async_playwright

from job_scan_mcp import config
from job_scan_mcp.repository import JobRepository
from job_scan_mcp.services.llm_factory import get_llm_for_stage

logger = logging.getLogger(__name__)


def build_base_cv_from_profile(profile) -> dict:
    """Convert a ParsedProfile into the base CV JSON shape used by the tailoring engine."""
    if profile is None:
        return {}
    contact = []
    if profile.email:
        contact.append(profile.email)
    return {
        "name": profile.name or "",
        "contact": contact,
        "summary": profile.summary or "",
        "experience": [
            {
                "title": job.title,
                "company": job.company,
                "date": job.date,
                "location": job.location,
                "bullets": list(job.bullets),
            }
            for job in (profile.experience or [])
        ],
        "education": [
            {"degree": edu, "institution": "", "year": ""}
            for edu in (profile.education or [])
        ],
        "skills": {"Technical": list(profile.skills or []), "Core Stack": list(profile.core_stack or [])},
    }


# =====================================================================
# Structured output schemas (each modified node carries metadata)
# =====================================================================
class TailoredBullet(BaseModel):
    text: str = Field(..., description="The bullet text (rewritten in XYZ format when modified)")
    modified: bool = Field(False, description="True when the LLM adapted/emphasized this bullet for the JD")
    match_reason: Optional[str] = Field(None, description="Why this change improves the match (references JD requirements)")


class TailoredJob(BaseModel):
    title: str
    company: str
    date: str = ""
    location: str = ""
    bullets: List[TailoredBullet] = Field(default_factory=list)


class TailoredEducation(BaseModel):
    degree: str
    institution: str = ""
    year: str = ""


class TailoredCV(BaseModel):
    name: str
    contact: List[str] = Field(default_factory=list)
    summary: str = ""
    experience: List[TailoredJob] = Field(default_factory=list)
    education: List[TailoredEducation] = Field(default_factory=list)
    skills: Dict[str, List[str]] = Field(default_factory=dict)


def _strip_placeholders(text: str) -> str:
    """Remove metric placeholders like '[X]%', '[X]+ TPS', '[Métrica]' so the CV is ready to send."""
    if not text:
        return text
    pat = re.compile(r"\[[^\]]{1,40}\]\s*(?:%|\+\s*[A-Za-zÁ-ú]+)?")
    cleaned = pat.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)              # collapse whitespace
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)  # attach dangling punctuation
    return cleaned.strip()


# =====================================================================
# Tool A: LLM inference / adaptation (reusable by the MCP tool and the pipeline)
# =====================================================================
async def build_tailored_cv_data(
    job_description_text: str,
    base_cv_json: dict,
    repo: JobRepository,
    llm: Any = None,
) -> dict:
    """Analyze the JD and rewrite the base CV to maximize alignment.

    Shared by the standalone `generate_tailored_cv` MCP tool and the background
    generation inside deep evaluation. Returns the tailored CV JSON where every
    modified bullet has ``modified: true`` and a ``match_reason``. Experience is
    never invented; it is only reframed, reordered and emphasized.
    """
    if not job_description_text or not job_description_text.strip():
        # No JD to adapt against: return the base CV unchanged (no fabricated changes).
        data = {
            "name": base_cv_json.get("name", ""),
            "contact": list(base_cv_json.get("contact") or []),
            "summary": base_cv_json.get("summary", ""),
            "experience": list(base_cv_json.get("experience") or []),
            "education": list(base_cv_json.get("education") or []),
            "skills": dict(base_cv_json.get("skills") or {}),
        }
        data["_meta"] = {
            "modified_bullets": 0,
            "total_bullets": sum(len(j.get("bullets", [])) for j in data["experience"]),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return data
    if not base_cv_json:
        raise ValueError("base_cv_json cannot be empty.")

    if llm is None:
        llm = await get_llm_for_stage("deep_evaluation", repo)
    structured_llm = llm.with_structured_output(TailoredCV)

    import json as _json
    base_json = _json.dumps(base_cv_json, ensure_ascii=False)[:8000]

    prompt = (
        "You are an elite Senior Technical Recruiter specialized in US Big Tech (Amazon, Meta, Netflix) "
        "rewriting CVs to L5/Senior engineer standards, ATS-optimized and ready for a single-column "
        "Harvard/ATS-friendly HTML template.\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{job_description_text[:8000]}\n\n"
        "=== BASE CV (JSON) ===\n"
        f"{base_json}\n\n"
        "STRICT RULES:\n"
        "1. NEVER invent experience, employers, dates or skills. Only reframe, reorder, emphasize and reword what exists.\n"
        "2. PRESERVE EVERY role in the base CV - never omit or drop any experience entry (e.g. keep the Dematic "
        "Commissioning Engineer role so continuous tenure ~6.8 years is reflected). Do not change employers or date spans.\n"
        "3. DATES: unify strictly to 'Mon YYYY - Mon YYYY' or 'Mon YYYY - Present' (e.g. 'Sep 2025 - Present', 'Jul 2024 - Sep 2025').\n"
        "4. BULLETS: always use XYZ format (Accomplished X, measured by Y, by doing Z). NEVER use "
        "placeholders like '[X]%', '[X]+ TPS' or '[Métrica]' - write complete, ready-to-send bullets "
        "using the concrete facts available. Every bullet must be a bullet item, never free text.\n"
        "5. Rewrite bullets that address JD requirements; set 'modified': true with a 'match_reason' referencing the JD "
        "requirement. Keep bullets that still fit as-is with 'modified': false and no match_reason.\n"
        "6. HEADER: format location as 'Mexico City, MX (Open to US Relocation / TN Visa Eligible)'. Include LinkedIn and "
        "GitHub placeholders in contact.\n"
        "7. SKILLS: restructure into exactly 3 categories: 'Languages & Frameworks', 'Cloud & Infrastructure', 'Architecture'.\n"
        "8. Adjust the summary to echo the JD's keywords, and return the complete CV structure (name, contact, summary, "
        "experience, education, skills).\n"
        "Output the JSON object conforming to the provided schema."
    )

    try:
        result: TailoredCV = await structured_llm.ainvoke(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM failed to tailor CV: {e}") from e

    # Ensure metadata integrity on every bullet + strip any metric placeholders (ready to send)
    for job in result.experience:
        for bullet in job.bullets:
            bullet.text = _strip_placeholders(bullet.text)
            if bullet.modified and not bullet.match_reason:
                bullet.match_reason = "Reworded to emphasize alignment with the job description."

    data = result.model_dump()
    modified_count = sum(
        1 for job in data.get("experience", []) for b in job.get("bullets", []) if b.get("modified")
    )
    data["_meta"] = {
        "modified_bullets": modified_count,
        "total_bullets": sum(len(j.get("bullets", [])) for j in data.get("experience", [])),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return data


# =====================================================================
# Preview HTML (side-panel style page with highlight engine)
# =====================================================================
def render_preview_html(tailored_cv: dict, preview_path: Optional[Path] = None) -> Path:
    """Render a self-contained preview page embedding the tailored CV."""
    from jinja2 import Environment, FileSystemLoader

    if preview_path is None:
        preview_dir = config.DATA_DIR / "cv_tailor"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"cv-tailor-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"

    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent.parent / "templates")))
    template = env.get_template("cv_tailor_preview.html")
    html = template.render(
        tailored=tailored_cv,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(html, encoding="utf-8")
    return preview_path


# =====================================================================
# Tool B: Playwright PDF export (reference ATS template preserved)
# =====================================================================
PDF_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @page { margin: 0.4in; }
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111; line-height: 1.25; font-size: 10pt; }
        h1 { font-size: 18pt; text-align: center; margin: 0 0 2px; font-weight: normal; }
        .contact-info { text-align: center; font-size: 9pt; margin: 0 0 10px; color: #333; }
        .section-title { font-size: 11pt; text-transform: uppercase; border-bottom: 1px solid #000; margin-top: 8px; margin-bottom: 6px; font-weight: bold; }
        .item-header { display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 1px; }
        .item-subheader { display: flex; justify-content: space-between; font-style: italic; margin-bottom: 2px; font-size: 9.5pt;}
        ul { margin-top: 0; padding-left: 18px; margin-bottom: 8px; }
        li { margin-bottom: 2px; }
        .summary { margin: 0 0 8px; font-size: 9.5pt; }
        .skills-container { margin-bottom: 8px; }
        .skill-row { margin-bottom: 2px; font-size: 9.5pt; }
        .bold { font-weight: bold; }
    </style>
</head>
<body>
    <h1>{{ cv.name or '' }}</h1>
    <div class="contact-info">{{ cv.contact | join(' &nbsp;|&nbsp; ') | safe }}</div>
    <div class="summary">{{ cv.summary or '' }}</div>

    <div class="section-title">Experience</div>
    {% for job in (cv.experience or []) %}
    <div>
        <div class="item-header"><span>{{ job.title }}</span><span>{{ job.date }}</span></div>
        <div class="item-subheader"><span>{{ job.company }}</span><span>{{ job.location }}</span></div>
        <ul>{% for bullet in (job.bullets or []) %}<li>{{ bullet.text if bullet is mapping else bullet }}</li>{% endfor %}</ul>
    </div>
    {% endfor %}

    {% if cv.education %}
    <div class="section-title">Education</div>
    {% for edu in cv.education %}
    <div>
        <div class="item-header"><span>{{ edu.degree or '' }}</span><span>{{ edu.year or '' }}</span></div>
        <div class="item-subheader"><span>{{ edu.institution or '' }}</span><span></span></div>
    </div>
    {% endfor %}
    {% endif %}

    {% if cv.skills %}
    <div class="section-title">Skills</div>
    <div class="skills-container">
        {% for group, items in cv.skills.items() %}
        <div class="skill-row"><span class="bold">{{ group }}:</span> {{ (items or []) | join(', ') }}</div>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
"""


def _safe_filename(file_name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', file_name or "").strip(" _")
    name = name or "tailored-cv"
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name


async def export_cv_to_pdf(tailored_cv_data: dict, file_name: str) -> str:
    """Render the validated tailored CV to PDF using Playwright (ATS-optimized template)."""
    if not tailored_cv_data:
        raise ValueError("tailored_cv_data cannot be empty.")

    template = Template(PDF_TEMPLATE)
    rendered_html = template.render(cv=tailored_cv_data)

    tmp_dir = config.DATA_DIR / "cv_tailor"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    html_path = tmp_dir / f"_tmp_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.html"
    html_path.write_text(rendered_html, encoding="utf-8")

    out_dir = config.DATA_DIR / "cv_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{_safe_filename(file_name)}.pdf"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page()
            await page.goto(html_path.as_uri())
            await page.pdf(
                path=str(output_path),
                format="Letter",
                print_background=True,
                margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"},
            )
        finally:
            await browser.close()

    html_path.unlink(missing_ok=True)
    logger.info("CV exported to %s", output_path)
    return str(output_path)
