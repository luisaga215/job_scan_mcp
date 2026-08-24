import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Field, SQLModel

# =====================================================================
# Pydantic Schemas for JSON Structured Outputs & API payloads
# =====================================================================

class ParsedProfile(BaseModel):
    name: Optional[str] = PydanticField(None, description="The candidate's full name")
    email: Optional[str] = PydanticField(None, description="The candidate's email address")
    skills: List[str] = PydanticField(default_factory=list, description="List of technical skills and tools")
    core_stack: List[str] = PydanticField(default_factory=list, description="Primary languages, frameworks, or cloud providers (e.g., Python, AWS)")
    experience_years: float = PydanticField(0.0, description="Total years of professional software engineering experience")
    seniority_level: str = PydanticField("Mid", description="Estimated seniority level (e.g., Junior, Mid, Senior, Lead, Principal, Staff)")
    education: List[str] = PydanticField(default_factory=list, description="Degrees, universities, or notable certifications")
    summary: str = PydanticField("", description="A short professional summary of the candidate's technical profile")


class DeepEvaluationResult(BaseModel):
    fit_score: int = PydanticField(..., ge=0, le=100, description="Overall match percentage (0-100) between the job and user profile")
    interview_probability: int = PydanticField(..., ge=0, le=100, description="Estimated chance of getting an initial call (0-100)")
    core_stack_overlap: List[str] = PydanticField(default_factory=list, description="Specific technologies in the job that match the user's stack")
    true_seniority_alignment: str = PydanticField(..., description="Evaluation of whether the role matches user's actual seniority, identifying disguised roles")
    red_flags: List[str] = PydanticField(default_factory=list, description="Hidden warnings (e.g., 'wear many hats', uncompensated on-call)")
    application_friction: str = PydanticField(..., description="Low, Medium, or High (based on application steps, take-homes, round count)")
    pros: List[str] = PydanticField(default_factory=list, description="Strongest alignment points of the role")
    cons: List[str] = PydanticField(default_factory=list, description="Weakest alignment points or concerns of the role")


# =====================================================================
# SQLModel Database Models
# =====================================================================

class LLMConfig(SQLModel, table=True):
    __tablename__: str = "llm_configs"
    
    stage: str = Field(primary_key=True, description="The stage this config applies to (e.g., fast_screening, deep_evaluation)")
    provider: str = Field(description="LLM provider name (e.g., openai, ollama, gemini)")
    model: str = Field(description="Name of the model to use (e.g., gpt-4o-mini, llama3)")
    base_url: Optional[str] = Field(default=None, description="Optional custom base URL for the API endpoint")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class PipelineMeta(SQLModel, table=True):
    __tablename__: str = "pipeline_meta"

    key: str = Field(primary_key=True, description="Metadata key (e.g., 'last_search')")
    value_json: str = Field(default="{}", description="JSON-serialized metadata value")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class UserProfile(SQLModel, table=True):
    __tablename__: str = "user_profiles"
    
    id: str = Field(default="default", primary_key=True)
    raw_cv_text: str = Field(description="Raw extracted text of the CV")
    parsed_profile_json: str = Field(description="JSON serialized string of ParsedProfile Pydantic model")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    @property
    def profile(self) -> ParsedProfile:
        return ParsedProfile.model_validate_json(self.parsed_profile_json)

    @profile.setter
    def profile(self, val: ParsedProfile) -> None:
        self.parsed_profile_json = val.model_dump_json()


class Job(SQLModel, table=True):
    __tablename__: str = "jobs"

    id: str = Field(primary_key=True, description="Deterministic SHA-256 hash of lowercase(company + title + location)")
    title: str = Field(index=True)
    company: str = Field(index=True)
    location: str = Field(index=True)
    description: str
    job_url: str
    salary_min: Optional[float] = Field(default=None)
    salary_max: Optional[float] = Field(default=None)
    currency: Optional[str] = Field(default=None)
    is_remote: Optional[bool] = Field(default=None, index=True)
    date_posted: Optional[str] = Field(default=None)

    # Visa / relocation eligibility signals (computed by job_service.analyze_visa_fit)
    sponsorship: Optional[bool] = Field(default=None, index=True, description="Job mentions visa sponsorship availability (H-1B, TN, etc.)")
    relocation_support: Optional[bool] = Field(default=None, index=True, description="Job mentions relocation assistance or package")
    us_eligible: Optional[bool] = Field(default=None, description="No explicit exclusion of candidates who need sponsorship")
    visa_keywords: Optional[str] = Field(default=None, description="Matched visa/relocation keyword signals for transparency")

    # User-managed application status (apply | applied | interview | rejected)
    application_status: Optional[str] = Field(default=None, index=True, description="User-managed application state for the kanban workflow")
    application_status_updated_at: Optional[datetime] = Field(default=None, description="When the user last updated the application status")
    
    # State Flow: PENDING_SCREENING -> RELEVANT / REJECTED -> EVALUATED
    state: str = Field(default="PENDING_SCREENING", index=True)
    screening_reason: Optional[str] = Field(default=None)
    
    # Deep evaluation metrics (populated when state is EVALUATED)
    fit_score: Optional[int] = Field(default=None, index=True)
    interview_probability: Optional[int] = Field(default=None)
    core_stack_overlap_json: Optional[str] = Field(default=None, description="JSON string list of core stack overlaps")
    true_seniority_alignment: Optional[str] = Field(default=None)
    red_flags_json: Optional[str] = Field(default=None, description="JSON string list of red flags")
    application_friction: Optional[str] = Field(default=None)
    pros_json: Optional[str] = Field(default=None, description="JSON string list of pros")
    cons_json: Optional[str] = Field(default=None, description="JSON string list of cons")
    evaluated_at: Optional[datetime] = Field(default=None)

    # Core stack overlap helper
    @property
    def core_stack_overlap(self) -> List[str]:
        return json.loads(self.core_stack_overlap_json) if self.core_stack_overlap_json else []

    @core_stack_overlap.setter
    def core_stack_overlap(self, val: List[str]) -> None:
        self.core_stack_overlap_json = json.dumps(val)

    # Red flags helper
    @property
    def red_flags(self) -> List[str]:
        return json.loads(self.red_flags_json) if self.red_flags_json else []

    @red_flags.setter
    def red_flags(self, val: List[str]) -> None:
        self.red_flags_json = json.dumps(val)

    # Pros helper
    @property
    def pros(self) -> List[str]:
        return json.loads(self.pros_json) if self.pros_json else []

    @pros.setter
    def pros(self, val: List[str]) -> None:
        self.pros_json = json.dumps(val)

    # Cons helper
    @property
    def cons(self) -> List[str]:
        return json.loads(self.cons_json) if self.cons_json else []

    @cons.setter
    def cons(self, val: List[str]) -> None:
        self.cons_json = json.dumps(val)
