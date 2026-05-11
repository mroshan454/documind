from app.services.query_service import retrieve_chunks
from app.services.query_service import query_documents

result = query_documents("What is Multi-Head Attention?", k=3)

print("=" * 60)
print("ANSWER:")
print(result["answer"])
print("=" * 60) 
print("\nSOURCES:")
for s in result["sources"]:
    print(f" - {s['source']} , page {s['page']}")