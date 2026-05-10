from app.services.ingestion_service import ingest_document , _embedding_model
from langchain_chroma import Chroma 

#result = ingest_document("test.pdf")
#print(f"Ingestion Complete!")
#print(f"Pages processed: {result['pages']}")
#print(f"Chunks Stored: {result['chunks']}")

#Step 2 : Open The existing ChromaDB and search it
vector_store = Chroma(
    collection_name="documind",
    embedding_function=_embedding_model,
    persist_directory="./chroma_db"
)

#Step 3: Try a similarity Search 
query = "What is the Transformer Architecture?"
results = vector_store.similarity_search(query,k=3)

print(f"Query: {query}\n")
for i , doc in enumerate(results, 1):
    print(f"----Result {i}---")
    print(f"Page: {doc.metadata.get('page_label','?')}")
    print(f"Text: {doc.page_content[:200]}....")
    print()