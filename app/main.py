from fastapi import FastAPI 

app = FastAPI() 

@app.get("/health")
def health_check(): 
    return {"status": "ok"} 

@app.post("/ingest")
def ingest_document():
    return {"message": "ingest endpoint - not implemented yet"}

@app.post("/query")
def query_document():
    return {"message": "query endpoint - not implemented yet"}

    