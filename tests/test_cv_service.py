import os
import tempfile
import pytest
from unittest.mock import AsyncMock, patch

from job_scan_mcp.models import ParsedProfile
from job_scan_mcp.services.cv_service import (
    extract_text_from_md,
    parse_and_sync_cv
)
from conftest import MockStructuredLLM

def test_extract_text_from_md():
    """Test reading raw text from a Markdown file."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
        f.write("# John Doe\nPython developer with 5 years experience.")
        temp_name = f.name
        
    try:
        text = extract_text_from_md(temp_name)
        assert "John Doe" in text
        assert "5 years experience" in text
    finally:
        os.remove(temp_name)

@pytest.mark.asyncio
@patch("job_scan_mcp.services.cv_service.extract_text_from_pdf")
@patch("job_scan_mcp.services.cv_service.get_llm_for_stage")
async def test_parse_and_sync_cv_pdf(mock_get_llm, mock_extract_pdf, test_repo):
    """Test parsing a PDF resume and saving structured profile in database."""
    # 1. Setup mock PDF text
    mock_extract_pdf.return_value = "Jane Doe\nPrincipal Engineer\nPython, AWS, 10 years experience."
    
    # 2. Setup mock LLM structured result
    expected_profile = ParsedProfile(
        name="Jane Doe",
        email="jane@example.com",
        skills=["Python", "AWS"],
        core_stack=["Python", "AWS"],
        experience_years=10.0,
        seniority_level="Principal",
        education=["B.S. Computer Science"],
        summary="Experienced engineer specializing in python and cloud architectures."
    )
    mock_llm = MockStructuredLLM(expected_profile)
    mock_get_llm.return_value = mock_llm
    
    # 3. Call service
    result = await parse_and_sync_cv("mock_resume.pdf", test_repo)
    
    # 4. Verify assertions
    assert result.id == "default"
    assert result.raw_cv_text == mock_extract_pdf.return_value
    assert result.profile.name == "Jane Doe"
    assert result.profile.experience_years == 10.0
    assert "Python" in result.profile.skills
    
    # 5. Verify database persistence
    db_profile = await test_repo.get_user_profile()
    assert db_profile is not None
    assert db_profile.profile.name == "Jane Doe"

@pytest.mark.asyncio
async def test_parse_and_sync_cv_unsupported_format(test_repo):
    """Test that unsupported formats raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        await parse_and_sync_cv("resume.docx", test_repo)
