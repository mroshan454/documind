from unittest.mock import AsyncMock , patch 
from fastapi.testclient import TestClient 
from app.main import app 

client = TestClient(app)

def test_health_endpoint_returns_ok():
    """GET /health should return 200 and a JSON status."""
    response = client.get("/health")
    assert response.status_code == 200 
    assert response.json() == {"status":"ok"}

def test_query_endpoint_returns_answer_and_sources():
    """POST /query should return an answer with sources."""

    fake_response = AsyncMock()
    fake_response.content = "Multi-head attention is a mechanism..."

    with patch("app.services.query_service._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)

        response = client.post(
            "/query",
            json={"question":"What is multi-head attention?","k":3},
        )
    
    assert response.status_code == 200 
    body = response.json()
    assert "answer" in body 
    assert "sources" in body 
    assert body["answer"] == "Multi-head attention is a mechanism..."