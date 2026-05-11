from fastapi import FastAPI 
from pydantic import BaseModel 
from app.services.ingestion_service import ingest_document
from app.services.query_service import query_documents

app = FastAPI() 

class QueryRequest(BaseModel):
    question: str 
    k : int = 3 

@app.get("/health")
def health_check(): 
    return {"status": "ok"} 

@app.post("/ingest")
def ingest_endpoint():
    result = ingest_document("test.pdf")
    return result

@app.post("/query")
def query_endpoint(request: QueryRequest):
    return query_documents(request.question,k=request.k)