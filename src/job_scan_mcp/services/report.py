import json
import logging
import re
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from jinja2 import Environment, FileSystemLoader

from job_scan_mcp.config import REPORTS_DIR
from job_scan_mcp.repository import JobRepository

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
MAX_REPORTS = 30

# =====================================================================
# Skill-gap analysis helpers (deterministic, no extra LLM calls)
# =====================================================================
TECH_KEYWORDS = [
    "java", "spring", "spring boot", "hibernate", "postgres", "postgresql", "sql server",
    "aws", "sqs", "sns", "rds", "ec2", "s3", "lambda", "kubernetes", "k8s", "docker",
    "kafka", "redis", "mongodb", "mysql", "cassandra", "terraform", "python", "go ",
    "golang", "rust", "c++", "typescript", "javascript", "react", "angular", "node.js",
    "graphql", "grpc", "microservices", "distributed systems", "git", "ci/cd", "linux",
    "spark", "flink", "hadoop", "elasticsearch", "snowflake", "datadog", "prometheus",
    "grafana", "gcp", "azure", "cloud", "s3", "dynamodb",
]

# Transferability families: a gap term is "yellow" (transferable) if the candidate
# already knows a sibling technology from the same family.
TRANSFER_FAMILIES = {
    "aws": ["gcp", "azure", "cloud"],
    "gcp": ["aws", "azure", "cloud"],
    "azure": ["aws", "gcp", "cloud"],
    "cloud": ["aws", "gcp", "azure"],
    "kubernetes": ["docker", "container"],
    "react": ["typescript", "javascript", "frontend"],
    "angular": ["typescript", "javascript", "frontend"],
    "postgres": ["mysql", "sql", "database"],
    "mysql": ["postgres", "sql", "database"],
    "python": ["java"],
    "java": ["python"],
    "spark": ["hadoop", "flink"],
    "flink": ["spark", "hadoop"],
}

def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def compute_skill_chips(job_description: str, profile_skills: list, profile_core_stack: list) -> list:
    """Classify tech keywords found in the job description against the candidate profile.

    Returns a list of {name, status} where status is 'match' (green), 'transferable'
    (yellow) or 'gap' (red). Deterministic and capped for readability.
    """
    desc = _normalize(job_description)
    profile_terms = {_normalize(s) for s in (profile_skills or []) + (profile_core_stack or [])}

    chips = []
    for kw in TECH_KEYWORDS:
        token = kw.strip()
        if token and token in desc:
            if token in profile_terms:
                chips.append({"name": kw.strip(), "status": "match"})
            else:
                family = TRANSFER_FAMILIES.get(token, [])
                transferable = any(f in profile_terms for f in family)
                chips.append({"name": kw.strip(), "status": "transferable" if transferable else "gap"})
    # Deduplicate (e.g., 'postgres' and 'postgresql')
    seen = set()
    unique = []
    for chip in chips:
        key = chip["name"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(chip)
    # Order: matches first, then transferable, then gaps
    order = {"match": 0, "transferable": 1, "gap": 2}
    unique.sort(key=lambda c: (order.get(c["status"], 3), c["name"]))
    return unique[:8]


def _parse_posted_date(raw: str) -> date:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    # Try to parse "X days ago" / "X hours ago"
    raw_lower = raw.lower()
    for unit, days in (("day", 1), ("hour", 0), ("week", 7), ("month", 30)):
        try:
            if f"{unit} ago" in raw_lower:
                num = int("".join(ch for ch in raw_lower.split(" ago")[0].split()[-2:] if ch.isdigit()) or "0")
                return date.today() - date.timedelta(days=num * days)
        except (ValueError, IndexError):
            continue
    return None


def relative_date_str(posted: date) -> str:
    if posted is None:
        return "Unknown date"
    delta = (date.today() - posted).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta < 7:
        return f"{delta} days ago"
    if delta < 30:
        return f"{delta // 7} week(s) ago"
    return f"{delta // 30} month(s) ago"

def _load_manifest() -> dict:
    """Load the report history manifest, or return an empty structure."""
    manifest_path = REPORTS_DIR / MANIFEST_FILE
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt report manifest; starting fresh.")
    return {"latest_file": "index.html", "entries": []}


def _save_manifest(manifest: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_manifest_archived(filename: str, archived: bool) -> None:
    manifest = _load_manifest()
    for entry in manifest.get("entries", []):
        if entry.get("file") == filename:
            entry["archived"] = archived
            break
    _save_manifest(manifest)


def _replace_manifest_in_html(text: str, new_json: str) -> str:
    """Swap the embedded window.reportsManifest array in a snapshot HTML (bracket-aware)."""
    return _replace_js_json_in_html(text, "window.reportsManifest = ", new_json, "[", "]")


def _replace_current_report_in_html(text: str, new_json: str) -> str:
    """Swap the embedded window.currentReport object in a snapshot HTML (bracket-aware)."""
    return _replace_js_json_in_html(text, "window.currentReport = ", new_json, "{", "}")


def _replace_js_json_in_html(text: str, marker: str, new_json: str, open_ch: str, close_ch: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return text
    start = idx + len(marker)
    depth = 0
    i = start
    in_str = False
    esc = False
    end = len(text)
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        i += 1
    return text[:start] + new_json + text[end:]


def _sync_manifest_into_snapshots() -> None:
    """Refresh the embedded report history (and each snapshot's own entry) in every snapshot file."""
    manifest = _load_manifest()
    entries = manifest["entries"]
    by_file = {e["file"]: e for e in entries}
    list_json = json.dumps(entries, ensure_ascii=False)
    for f in REPORTS_DIR.glob("report-*.html"):
        try:
            text = f.read_text(encoding="utf-8")
            updated = _replace_manifest_in_html(text, list_json)
            own = by_file.get(f.name)
            if own:
                updated = _replace_current_report_in_html(updated, json.dumps(own, ensure_ascii=False))
            if updated != text:
                f.write_text(updated, encoding="utf-8")
        except OSError:
            logger.warning("Could not sync manifest into snapshot %s", f)


def archive_report_file(filename: str) -> dict:
    """Move a report snapshot to the archive folder (soft delete, reversible)."""
    archive_dir = REPORTS_DIR / "archive"
    src = REPORTS_DIR / filename
    if not src.exists():
        return {"ok": False, "error": f"Report file '{filename}' not found in {REPORTS_DIR}."}
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / filename
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    _set_manifest_archived(filename, True)
    logger.info("Archived report %s -> %s", filename, archive_dir)
    return {"ok": True, "archived": filename, "archive_dir": str(archive_dir)}


def restore_report_file(filename: str) -> dict:
    """Move a report snapshot back from the archive folder to the active reports dir."""
    archive_dir = REPORTS_DIR / "archive"
    src = archive_dir / filename
    if not src.exists():
        return {"ok": False, "error": f"Archived report '{filename}' not found in {archive_dir}."}
    dst = REPORTS_DIR / filename
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    _set_manifest_archived(filename, False)
    logger.info("Restored report %s", filename)
    return {"ok": True, "restored": filename, "reports_dir": str(REPORTS_DIR)}


async def _llm_config_summary(repo: JobRepository) -> dict:
    """Snapshot the LLM configuration in use for the report metadata."""
    from job_scan_mcp import config as cfg
    screening_cfg = await repo.get_llm_config("fast_screening")
    evaluation_cfg = await repo.get_llm_config("deep_evaluation")
    return {
        "screening_model": f"{screening_cfg.provider}/{screening_cfg.model}" if screening_cfg else cfg.DEFAULT_SCREENING_MODEL,
        "evaluation_model": f"{evaluation_cfg.provider}/{evaluation_cfg.model}" if evaluation_cfg else cfg.DEFAULT_EVALUATION_MODEL,
    }


def _clean_ai_summary(text: str) -> str:
    """Normalize an LLM summary into plain bullet points (strip markdown/bold/bullets)."""
    lines = []
    for ln in str(text).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^[-•*\d.\s]+", "", ln).strip()   # strip leading bullets / numbers
        ln = re.sub(r"\*\*|__|`|\*", "", ln).strip()   # remove markdown emphasis / code ticks
        if ln:
            lines.append(ln)
    return "\n".join(f"• {ln}" for ln in lines)[:700]


def _clean_manifest_summaries() -> None:
    """Retroactively strip markdown from every stored AI summary and re-sync snapshots."""
    manifest = _load_manifest()
    changed = False
    for entry in manifest.get("entries", []):
        if entry.get("ai_summary"):
            cleaned = _clean_ai_summary(entry["ai_summary"])
            if cleaned != entry["ai_summary"]:
                entry["ai_summary"] = cleaned
                changed = True
    if changed:
        _save_manifest(manifest)
        _sync_manifest_into_snapshots()


async def _generate_ai_summary(repo: JobRepository, evaluated_jobs: list, avg_fit: float, top_fit: float) -> Optional[str]:
    """Produce a short AI summary of what this report found (best-effort)."""
    if not evaluated_jobs:
        return None
    try:
        from job_scan_mcp.services.llm_factory import get_llm_for_stage
        llm = await get_llm_for_stage("deep_evaluation", repo)
        top = sorted(evaluated_jobs, key=lambda j: j.fit_score or 0, reverse=True)[:8]
        digest = "\n".join(
            f"- {j.title} @ {j.company} | fit {j.fit_score or 0} | prob {j.interview_probability or 0} | flags {len(j.red_flags)}"
            for j in top
        )
        prompt = (
            "You are a job-search analyst. Summarize what this report found in 3-4 short English bullet points: "
            "dominant role types, standout opportunities (high fit and/or low friction), and common concerns (red flags).\n"
            "Write PLAIN TEXT ONLY: one bullet per line, starting with '-', no markdown, no bold, no asterisks, no backticks.\n"
            f"Evaluated jobs: {len(evaluated_jobs)} | avg fit: {round(avg_fit, 1)}% | top fit: {top_fit}%\n"
            f"Top roles by fit:\n{digest}\n\nBullet points:"
        )
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return _clean_ai_summary(text)
    except Exception as e:
        logger.warning("AI summary generation failed: %s", e)
        return None

async def generate_report(repo: JobRepository) -> str:
    """Query jobs, compile metrics, render the Jinja2 HTML dashboard, and save a report snapshot.

    Writes BOTH the latest `index.html` and a timestamped `report-<ts>.html` snapshot, and
    updates `manifest.json` so the UI can list and navigate past reports.
    """
    # 1. Fetch all jobs + candidate profile (for skill-gap analysis)
    jobs = await repo.get_all_jobs()
    profile_record = await repo.get_user_profile()
    profile = profile_record.profile if profile_record else None
    profile_skills = profile.skills if profile else []
    profile_core_stack = profile.core_stack if profile else []
    profile_name = profile.name if profile else "no profile"
    
    # 2. Compute statistics
    total_jobs = len(jobs)
    counts = await repo.get_pipeline_counts()
    
    evaluated_jobs = [j for j in jobs if j.state == "EVALUATED"]
    total_evaluated = len(evaluated_jobs)
    
    avg_fit = 0.0
    top_fit = 0
    if total_evaluated > 0:
        avg_fit = sum(j.fit_score or 0 for j in evaluated_jobs) / total_evaluated
        top_fit = max((j.fit_score or 0) for j in evaluated_jobs)

    metrics = {
        "total_jobs": total_jobs,
        "pending_screening": counts.get("PENDING_SCREENING", 0),
        "relevant": counts.get("RELEVANT", 0),
        "rejected": counts.get("REJECTED", 0),
        "evaluated": total_evaluated,
        "average_fit": round(avg_fit, 1),
        "top_fit": top_fit,
    }

    # 3. Format job list for Jinja2/JSON inclusion
    today = date.today()
    formatted_jobs = []
    for j in jobs:
        posted = _parse_posted_date(j.date_posted)
        days_since = (today - posted).days if posted is not None else None
        formatted_jobs.append({
            "id": j.id,
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "description": j.description,
            "job_url": j.job_url,
            "salary_min": j.salary_min,
            "salary_max": j.salary_max,
            "currency": j.currency,
            "is_remote": j.is_remote,
            "date_posted": j.date_posted,
            "state": j.state,
            "screening_reason": j.screening_reason or "",
            "fit_score": j.fit_score or 0,
            "interview_probability": j.interview_probability or 0,
            "core_stack_overlap": j.core_stack_overlap,
            "true_seniority_alignment": j.true_seniority_alignment or "",
            "red_flags": j.red_flags,
            "application_friction": j.application_friction or "Medium",
            "pros": j.pros,
            "cons": j.cons,
            "evaluated_at": j.evaluated_at.isoformat() if j.evaluated_at else None,
            # Visa/relocation signals (already stored on the Job row)
            "sponsorship": j.sponsorship,
            "relocation_support": j.relocation_support,
            "us_eligible": j.us_eligible,
            "visa_keywords": j.visa_keywords or "",
            # User-managed application status (kanban)
            "application_status": j.application_status,
            "application_status_updated_at": j.application_status_updated_at.isoformat() if j.application_status_updated_at else None,
            # Derived UI helpers
            "days_since_posted": days_since,
            "relative_date": relative_date_str(posted),
            "is_new": days_since is not None and days_since <= 7,
            "skill_chips": compute_skill_chips(j.description, profile_skills, profile_core_stack),
            # Tailored CV generated by the pipeline (deep evaluation) or on demand
            "tailored_cv": json.loads(j.tailored_cv_json) if j.tailored_cv_json else None,
        })

    # 4. Render using Jinja2
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report_template.html")

    generated_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    llm_models = await _llm_config_summary(repo)

    # 5. Build report history entry + snapshot
    manifest = _load_manifest()
    _clean_manifest_summaries()  # strip markdown from older summaries retroactively
    manifest = _load_manifest()
    last_search = await repo.get_pipeline_meta("last_search") or {}
    search_context = {
        "queries": last_search.get("queries") or [],
        "locations": last_search.get("locations") or [],
        "is_remote": last_search.get("is_remote"),
        "min_salary": last_search.get("min_salary"),
        "require_visa_friendly": last_search.get("require_visa_friendly"),
        "ran_at": last_search.get("ran_at"),
    }
    ai_summary = await _generate_ai_summary(repo, evaluated_jobs, avg_fit, top_fit)

    entry = {
        "file": None,  # filled below (timestamped snapshot)
        "generated_at": generated_at_str,
        "total_jobs": total_jobs,
        "pending_screening": counts.get("PENDING_SCREENING", 0),
        "relevant": counts.get("RELEVANT", 0),
        "rejected": counts.get("REJECTED", 0),
        "evaluated": total_evaluated,
        "average_fit": round(avg_fit, 1),
        "profile": profile_name,
        "search": search_context,
        "ai_summary": ai_summary,
        **llm_models,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    html_content = template.render(
        jobs=formatted_jobs,
        metrics=metrics,
        generated_at=generated_at_str,
        profile_summary=profile.summary if profile else "",
        profile_core_stack=(profile.core_stack if profile else []) or [],
        reports_manifest=manifest["entries"],
        current_report=entry,
    )

    # 6. Write latest index.html + timestamped snapshot + manifest
    report_file = REPORTS_DIR / "index.html"
    report_file.write_text(html_content, encoding="utf-8")

    snapshot_file = REPORTS_DIR / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.html"
    snapshot_file.write_text(html_content, encoding="utf-8")

    entry["file"] = snapshot_file.name
    manifest["latest_file"] = "index.html"
    manifest["entries"].insert(0, entry)
    manifest["entries"] = manifest["entries"][:MAX_REPORTS]
    _save_manifest(manifest)

    # Every existing snapshot now points at the live history (old reports show new reports too)
    _sync_manifest_into_snapshots()

    logger.info(f"Report generated at: {report_file} (snapshot: {snapshot_file.name})")

    # 7. Return latest as file:// URI
    return report_file.as_uri()
