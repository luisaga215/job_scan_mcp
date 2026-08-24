import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from job_scan_mcp.models import LLMConfig, UserProfile, Job, ParsedProfile, PipelineMeta

class JobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_llm_config(self, stage: str) -> Optional[LLMConfig]:
        """Fetch LLM configuration for a specific pipeline stage."""
        statement = select(LLMConfig).where(LLMConfig.stage == stage)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def save_llm_config(self, stage: str, provider: str, model: str, base_url: Optional[str] = None) -> LLMConfig:
        """Save or override the LLM configuration for a stage."""
        config = await self.get_llm_config(stage)
        if config:
            config.provider = provider
            config.model = model
            config.base_url = base_url
            config.updated_at = datetime.utcnow()
        else:
            config = LLMConfig(
                stage=stage,
                provider=provider,
                model=model,
                base_url=base_url,
                updated_at=datetime.utcnow()
            )
            self.session.add(config)
        await self.session.flush()
        return config

    async def get_user_profile(self) -> Optional[UserProfile]:
        """Fetch the default candidate profile."""
        statement = select(UserProfile).where(UserProfile.id == "default")
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def save_user_profile(self, raw_cv_text: str, parsed_profile: ParsedProfile) -> UserProfile:
        """Save or update the single user profile in the database."""
        profile_record = await self.get_user_profile()
        if profile_record:
            profile_record.raw_cv_text = raw_cv_text
            profile_record.profile = parsed_profile
            profile_record.updated_at = datetime.utcnow()
        else:
            profile_record = UserProfile(
                id="default",
                raw_cv_text=raw_cv_text,
                updated_at=datetime.utcnow()
            )
            profile_record.profile = parsed_profile
            self.session.add(profile_record)
        await self.session.flush()
        return profile_record

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Fetch a job by its deterministic ID."""
        statement = select(Job).where(Job.id == job_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def save_job(self, job: Job) -> Job:
        """Save or update a job posting in the database."""
        existing = await self.get_job(job.id)
        if existing:
            # Update fields
            existing.title = job.title
            existing.company = job.company
            existing.location = job.location
            existing.description = job.description
            existing.job_url = job.job_url
            existing.salary_min = job.salary_min
            existing.salary_max = job.salary_max
            existing.currency = job.currency
            existing.is_remote = job.is_remote
            existing.date_posted = job.date_posted
            existing.state = job.state
            existing.screening_reason = job.screening_reason
            existing.sponsorship = job.sponsorship
            existing.relocation_support = job.relocation_support
            existing.us_eligible = job.us_eligible
            existing.visa_keywords = job.visa_keywords
            existing.application_status = job.application_status
            existing.application_status_updated_at = job.application_status_updated_at
            existing.fit_score = job.fit_score
            existing.interview_probability = job.interview_probability
            existing.core_stack_overlap_json = job.core_stack_overlap_json
            existing.true_seniority_alignment = job.true_seniority_alignment
            existing.red_flags_json = job.red_flags_json
            existing.application_friction = job.application_friction
            existing.pros_json = job.pros_json
            existing.cons_json = job.cons_json
            existing.evaluated_at = job.evaluated_at
            self.session.add(existing)
            await self.session.flush()
            return existing
        else:
            self.session.add(job)
            await self.session.flush()
            return job

    async def get_jobs_by_state(self, state: str, limit: Optional[int] = None) -> List[Job]:
        """Fetch jobs currently in a specific pipeline state."""
        statement = select(Job).where(Job.state == state)
        if limit:
            statement = statement.limit(limit)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_jobs_by_state_prioritized(self, state: str, limit: Optional[int] = None) -> List[Job]:
        """Fetch jobs in a state, prioritizing visa-friendly and relocation-supporting postings first."""
        statement = select(Job).where(Job.state == state)
        statement = statement.order_by(
            Job.sponsorship.desc().nullslast(),
            Job.relocation_support.desc().nullslast(),
        )
        if limit:
            statement = statement.limit(limit)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_all_jobs(self) -> List[Job]:
        """Fetch all jobs currently stored in the system."""
        statement = select(Job)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_pipeline_counts(self) -> Dict[str, int]:
        """Return the count of job records in each state."""
        counts = {
            "PENDING_SCREENING": 0,
            "RELEVANT": 0,
            "REJECTED": 0,
            "EVALUATED": 0
        }
        statement = select(Job.state, func.count(Job.id)).group_by(Job.state)
        result = await self.session.execute(statement)
        for state, count in result.all():
            if state in counts:
                counts[state] = count
        return counts

    async def set_application_status(self, job_id: str, status: str) -> Optional[Job]:
        """Persist the user-managed application status for a job."""
        allowed = {"apply", "applied", "interview", "rejected"}
        if status not in allowed:
            raise ValueError(f"Invalid application status '{status}'. Allowed: {sorted(allowed)}.")
        job = await self.get_job(job_id)
        if job is None:
            return None
        job.application_status = status
        job.application_status_updated_at = datetime.utcnow()
        await self.save_job(job)
        return job

    async def set_pipeline_meta(self, key: str, value: Any) -> PipelineMeta:
        """Upsert a JSON-serializable pipeline metadata value."""
        record = await self._get_pipeline_meta_record(key)
        if record:
            record.value_json = json.dumps(value, ensure_ascii=False)
            record.updated_at = datetime.utcnow()
        else:
            record = PipelineMeta(key=key, value_json=json.dumps(value, ensure_ascii=False))
            self.session.add(record)
        await self.session.flush()
        return record

    async def get_pipeline_meta(self, key: str) -> Optional[Any]:
        """Fetch a JSON-deserialized pipeline metadata value (or None)."""
        record = await self._get_pipeline_meta_record(key)
        if record is None:
            return None
        try:
            return json.loads(record.value_json)
        except (json.JSONDecodeError, TypeError):
            return None

    async def _get_pipeline_meta_record(self, key: str) -> Optional[PipelineMeta]:
        statement = select(PipelineMeta).where(PipelineMeta.key == key)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()
