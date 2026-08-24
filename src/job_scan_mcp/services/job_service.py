import hashlib
import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlmodel.ext.asyncio.session import AsyncSession
from jobspy import scrape_jobs
import pandas as pd

from job_scan_mcp.models import Job
from job_scan_mcp.repository import JobRepository

logger = logging.getLogger(__name__)

# =====================================================================
# Visa / Relocation keyword signals for US-oriented searches
# =====================================================================

SPONSORSHIP_POSITIVE_KEYWORDS = [
    "visa sponsorship",
    "visa sponsoring",
    "h-1b",
    "h1b",
    "h-1b visa",
    "tn visa",
    "tn status",
    "tn category",
    "work visa",
    "visa assistance",
    "visa support",
    "will sponsor",
    "sponsor visas",
    "sponsors visas",
    "sponsorship available",
    "sponsorship provided",
    "sponsorship offered",
    "provide sponsorship",
    "offers sponsorship",
    "green card sponsorship",
    "eb-2",
    "eb-3",
]

RELOCATION_POSITIVE_KEYWORDS = [
    "relocation assistance",
    "relocation package",
    "relocation support",
    "relocation reimbursement",
    "relocation benefits",
    "relocation bonus",
    "relocation provided",
    "relocation available",
    "relocation offered",
    "relo package",
    "relocation cost",
]

VISA_NEGATIVE_KEYWORDS = [
    "must be authorized to work",
    "must already be authorized",
    "work authorization",
    "authorized to work in the us",
    "authorized to work in the united states",
    "without sponsorship",
    "no sponsorship",
    "no visa sponsorship",
    "cannot sponsor",
    "cannot provide sponsorship",
    "does not sponsor",
    "doesn't sponsor",
    "unable to sponsor",
    "will not sponsor",
    "sponsorship is not",
    "sponsorship not available",
    "us citizens only",
    "u.s. citizens only",
    "us citizen or green card",
    "green card holders only",
    "permanent residents only",
    "us person",
]

RELOCATION_NEGATIVE_KEYWORDS = [
    "no relocation",
    "no relocation assistance",
    "no relocation package",
    "relocation not provided",
    "relocation not available",
    "no relocation support",
    "does not provide relocation",
    "cannot provide relocation",
]

US_LOCATION_PATTERN = re.compile(
    r"\b(united states|usa|u\.s\.|u s a|california|texas|washington|new york|"
    r"illinois|georgia|virginia|colorado|oregon|north carolina|massachusetts|"
    r"pennsylvania|florida|arizona|utah|austin|seattle|san francisco|new york city|"
    r"chicago|atlanta|denver|boston|miami|dallas|houston)\b",
    re.IGNORECASE,
)


def analyze_visa_fit(job_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a job dict for sponsorship / relocation / US-eligibility signals.

    Returns dict with boolean sponsorship, relocation_support, us_eligible and
    the matched positive keywords (for transparency).
    """
    location = (job_dict.get("location") or "").lower()
    description = (job_dict.get("description") or "").lower()
    text = f"{location}\n{description}"

    pos_sponsorship = [k for k in SPONSORSHIP_POSITIVE_KEYWORDS if k in text]
    pos_relocation = [k for k in RELOCATION_POSITIVE_KEYWORDS if k in text]
    neg_visa = [k for k in VISA_NEGATIVE_KEYWORDS if k in text]
    neg_relo = [k for k in RELOCATION_NEGATIVE_KEYWORDS if k in text]

    # Negative signals take precedence (e.g., "no visa sponsorship" still matches the
    # positive substring "visa sponsorship", but the intent is clearly negative).
    sponsorship = bool(pos_sponsorship) and not bool(neg_visa)
    relocation_support = bool(pos_relocation) and not bool(neg_relo)
    hostile = bool(neg_visa) or bool(neg_relo)

    return {
        "sponsorship": sponsorship,
        "relocation_support": relocation_support,
        "us_eligible": not hostile,
        "visa_keywords": ", ".join(pos_sponsorship + pos_relocation) or None,
    }

def generate_job_id(company: str, title: str, location: str) -> str:
    """Generate a deterministic SHA-256 hash for a job posting.
    This guarantees idempotency and ignores UTM tracking parameter variations.
    """
    company_clean = (company or "").strip().lower()
    title_clean = (title or "").strip().lower()
    location_clean = (location or "").strip().lower()
    key = f"{company_clean}|{title_clean}|{location_clean}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def apply_filters(
    job_dict: Dict[str, Any], 
    is_remote_filter: Optional[bool] = None, 
    min_salary_filter: Optional[float] = None,
    require_visa_friendly: Optional[bool] = None,
    visa_analysis: Optional[Dict[str, Any]] = None
) -> bool:
    """Apply deterministic Python-level filters. Tolerates None/missing fields by default."""
    
    # 1. Remote filtering
    is_remote = job_dict.get("is_remote")
    location = (job_dict.get("location") or "").lower()
    description = (job_dict.get("description") or "").lower()
    
    # Heuristic remote check
    seems_remote = is_remote is True or "remote" in location or "remote" in description or "work from home" in description
    
    if is_remote_filter is True:
        # User wants remote only: keep if remote flag is True or heuristics match
        if not seems_remote:
            return False
    elif is_remote_filter is False:
        # User wants on-site/hybrid only: reject if explicitly marked remote
        if is_remote is True:
            return False

    # 2. Salary filtering
    salary_min = job_dict.get("salary_min")
    salary_max = job_dict.get("salary_max")
    
    # If filter is set and salary data exists, reject if max or min is below threshold.
    # If salary data is None, we keep the job (tolerating None fields).
    if min_salary_filter is not None:
        has_salary_info = salary_min is not None or salary_max is not None
        if has_salary_info:
            val_to_check = salary_max if salary_max is not None else salary_min
            if val_to_check and val_to_check < min_salary_filter:
                return False

    # 3. Strict visa-friendly filter: keep ONLY postings with explicit sponsorship or relocation signals
    if require_visa_friendly:
        if visa_analysis is None:
            visa_analysis = analyze_visa_fit(job_dict)
        if not (visa_analysis["sponsorship"] or visa_analysis["relocation_support"]):
            return False
                
    return True

async def fetch_and_save_jobs(
    repo: JobRepository,
    queries: List[str],
    locations: List[str],
    max_pool_size: int = 150,
    is_remote: Optional[bool] = None,
    min_salary: Optional[float] = None,
    require_visa_friendly: Optional[bool] = None
) -> Dict[str, Any]:
    """Scrape job postings, apply Python-level filters, and persist non-duplicates as PENDING_SCREENING.

    If require_visa_friendly is True, only postings with explicit sponsorship or relocation
    signals are kept (strict US visa-oriented search).
    """
    all_scraped_jobs: List[Dict[str, Any]] = []
    
    # To avoid excessive calls, allocate results wanted per query/location combination
    combinations = [(q, l) for q in queries for l in locations]
    if not combinations:
        return {"error": "No queries or locations provided."}
        
    results_per_combination = max(5, int(max_pool_size / len(combinations)))
    
    for query, loc in combinations:
        logger.info(f"Scraping jobs for query='{query}', location='{loc}'")
        try:
            # Run jobspy scrape in a separate thread if it is synchronous
            # We call scrape_jobs. Note: jobspy returns a pandas DataFrame.
            df = scrape_jobs(
                site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
                search_term=query,
                location=loc,
                results_wanted=results_per_combination,
                hours_old=72,
                country_indeed='usa'
            )
            
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    # Map jobspy columns
                    salary_min = row.get("min_amount") if pd.notna(row.get("min_amount")) else None
                    salary_max = row.get("max_amount") if pd.notna(row.get("max_amount")) else None
                    
                    # Convert pandas NaNs to standard python None
                    job_data = {
                        "title": str(row.get("title")) if pd.notna(row.get("title")) else "",
                        "company": str(row.get("company")) if pd.notna(row.get("company")) else "",
                        "location": str(row.get("location")) if pd.notna(row.get("location")) else "",
                        "description": str(row.get("description")) if pd.notna(row.get("description")) else "",
                        "job_url": str(row.get("job_url")) if pd.notna(row.get("job_url")) else "",
                        "salary_min": float(salary_min) if salary_min is not None else None,
                        "salary_max": float(salary_max) if salary_max is not None else None,
                        "currency": str(row.get("currency")) if pd.notna(row.get("currency")) else None,
                        "is_remote": bool(row.get("is_remote")) if pd.notna(row.get("is_remote")) else None,
                        "date_posted": str(row.get("date_posted")) if pd.notna(row.get("date_posted")) else None,
                    }
                    
                    if job_data["title"] and job_data["company"]:
                        all_scraped_jobs.append(job_data)
        except Exception as e:
            logger.warning(f"Failed to scrape combination query={query}, loc={loc}: {str(e)}")
            # Continue to other combinations so a single failure doesn't crash the pipeline
            continue

    total_scraped = len(all_scraped_jobs)
    filtered_jobs: List[Dict[str, Any]] = []
    
    # Process unique jobs (dedup based on deterministic hash)
    seen_hashes = set()
    for job_data in all_scraped_jobs:
        job_hash = generate_job_id(job_data["company"], job_data["title"], job_data["location"])
        if job_hash in seen_hashes:
            continue
        seen_hashes.add(job_hash)
        
        # Apply filters
        visa_analysis = analyze_visa_fit(job_data)
        if apply_filters(job_data, is_remote, min_salary, require_visa_friendly, visa_analysis):
            job_data["id"] = job_hash
            job_data["visa_analysis"] = visa_analysis
            filtered_jobs.append(job_data)

    total_filtered = len(filtered_jobs)
    new_jobs_saved = 0
    updated_jobs = 0
    visa_friendly_saved = 0

    # Persist the search context so reports can show what was searched
    await repo.set_pipeline_meta("last_search", {
        "queries": queries,
        "locations": locations,
        "max_pool_size": max_pool_size,
        "is_remote": is_remote,
        "min_salary": min_salary,
        "require_visa_friendly": require_visa_friendly,
        "ran_at": datetime.now().isoformat(timespec="seconds"),
    })
    
    # Save to database
    for job_dict in filtered_jobs:
        visa = job_dict.get("visa_analysis", {})
        # Check if already exists in DB
        existing = await repo.get_job(job_dict["id"])
        if not existing:
            new_job = Job(
                id=job_dict["id"],
                title=job_dict["title"],
                company=job_dict["company"],
                location=job_dict["location"],
                description=job_dict["description"],
                job_url=job_dict["job_url"],
                salary_min=job_dict["salary_min"],
                salary_max=job_dict["salary_max"],
                currency=job_dict["currency"],
                is_remote=job_dict["is_remote"],
                date_posted=job_dict["date_posted"],
                state="PENDING_SCREENING",
                sponsorship=visa.get("sponsorship"),
                relocation_support=visa.get("relocation_support"),
                us_eligible=visa.get("us_eligible"),
                visa_keywords=visa.get("visa_keywords"),
            )
            await repo.save_job(new_job)
            new_jobs_saved += 1
            if visa.get("sponsorship") or visa.get("relocation_support"):
                visa_friendly_saved += 1
        else:
            # Refresh visa signals on re-fetches (backfill for rows created pre-analysis)
            changed = False
            for field, value in (
                ("sponsorship", visa.get("sponsorship")),
                ("relocation_support", visa.get("relocation_support")),
                ("us_eligible", visa.get("us_eligible")),
                ("visa_keywords", visa.get("visa_keywords")),
            ):
                if getattr(existing, field) != value:
                    setattr(existing, field, value)
                    changed = True
            if changed:
                await repo.save_job(existing)
                updated_jobs += 1
                if visa.get("sponsorship") or visa.get("relocation_support"):
                    visa_friendly_saved += 1
            
    return {
        "total_scraped": total_scraped,
        "total_filtered": total_filtered,
        "new_jobs_saved": new_jobs_saved,
        "updated_jobs": updated_jobs,
        "visa_friendly_saved": visa_friendly_saved
    }
