import shutil
from pathlib import Path 
from fastapi import FastAPI , UploadFile , File , HTTPException
from pydantic import BaseModel 
from app.services.ingestion_service import ingest_document
from app.services.query_service import query_documents
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("documind.api")

app = FastAPI() 

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    question: str 
    k : int = 3 

@app.get("/health")
def health_check(): 
    logger.info("health_check_called")
    return {"status": "ok"} 

@app.post("/ingest")
def ingest_endpoint(file: UploadFile = File(...)):
    logger.info("ingest_started",filename=file.filename)

    if not file.filename.endswith(".pdf"):
        logger.warning("ingest_rejected",filename=file.filename, reason="not_a_pdf")
        raise HTTPException(status_code=400, detail="Only PDF Files are accepted")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path,'wb') as f:
        shutil.copyfileobj(file.file,f)
    
    result = ingest_document(str(file_path))

    logger.info(
        "ingest_complete",
        filename=file.filename,
        pages=result["pages"],
        chunks=result["chunks"],
        file_hash=result["file_hash"],
    )
    return {
        "filename":file.filename, **result}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    logger.info("query_started",question=request.question, k=request.k)
    result = query_documents(request.question,k=request.k)
    logger.info(
        "query_complete",
        question=request.question,
        num_sources=len(result["sources"]),
    )
    return result 