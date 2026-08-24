import os
import pypdf
from typing import Optional
from job_scan_mcp.models import ParsedProfile, UserProfile
from job_scan_mcp.repository import JobRepository
from job_scan_mcp.services.llm_factory import get_llm_for_stage

def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    text_content = []
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file: {str(e)}")
        
    return "\n".join(text_content)

def extract_text_from_md(file_path: str) -> str:
    """Extract plain text from a Markdown/Text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

async def parse_and_sync_cv(file_path: str, repo: JobRepository) -> UserProfile:
    """Parse a CV file, extract its profile structure using an LLM, and save it to the DB."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif ext in [".md", ".txt"]:
        raw_text = extract_text_from_md(file_path)
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'. Only .pdf, .md, and .txt are supported.")
        
    if not raw_text.strip():
        raise ValueError("The extracted CV text is empty. Please provide a valid CV.")
        
    # Get LLM instance for parsing (using fast_screening stage LLM)
    llm = await get_llm_for_stage("fast_screening", repo)
    
    # Extract structured profile using langchain's with_structured_output
    structured_llm = llm.with_structured_output(ParsedProfile)
    
    prompt = (
        "You are an expert HR and recruitment assistant. Parse the following candidate resume text "
        "and extract a structured profile with candidate details, skills, core tech stack, "
        "experience years, education, estimated seniority, and a brief professional summary.\n\n"
        f"Resume text:\n{raw_text}"
    )
    
    try:
        parsed_profile: ParsedProfile = await structured_llm.ainvoke(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM failed to extract structured profile: {str(e)}")
        
    # Save user profile in database
    user_profile = await repo.save_user_profile(raw_text, parsed_profile)
    return user_profile
