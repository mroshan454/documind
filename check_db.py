from app.services.ingestion_service import _embedding_model
from langchain_chroma import Chroma

vs = Chroma(
    collection_name="documind",
    embedding_function=_embedding_model,
    persist_directory="./chroma_db",
)
print(f"Total chunks in DB: {vs._collection.count()}")