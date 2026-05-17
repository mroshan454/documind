from unittest.mock import AsyncMock, patch 
import pytest 
from app.services.query_service import call_llm
from app.core.exceptions import LLMUnavailableException

@pytest.mark.asyncio
async def test_call_llm_returns_response_content():
    """call_llm should return the .content of the LLM's response."""

    # Create a mock response object with a .content attribute 
    fake_response = AsyncMock()
    fake_response.content = "This is a fake LLM Answer."

    # Replace the real LLM with a fake one 
    with patch("app.services.query_service._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)

        result = await call_llm("Any Prompt")

    assert result == "This is a fake LLM Answer."

@pytest.mark.asyncio 
async def test_call_llm_raises_llm_unavailable_on_openai_failure():
    """When the OpenAI call fails, call_llm should raise LLMUnavailableException."""

    with patch("app.services.query_service._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("OpenAI is down"))

        with pytest.raises(LLMUnavailableException):
            await call_llm("any prompt")