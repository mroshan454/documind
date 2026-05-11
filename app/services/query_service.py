import os 
from dotenv import load_dotenv 
from langchain_chroma import Chroma 
from app.services.ingestion_service import _embedding_model, CHROMA_DIR 
from langchain_openai import ChatOpenAI

load_dotenv() # Reads .env file into environement variable 

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

def retrieve_chunks(query:str , k: int = 3):
    """Embed the query and retrieve top-k similar chunks from ChromaDB."""
    vector_store = Chroma(
        collection_name="documind",
        embedding_function=_embedding_model,
        persist_directory=CHROMA_DIR,
    )

    results = vector_store.similarity_search(query,k=k)
    return results

def build_prompt(query:str , chunks):
    """Build a RAG Prompt: Instructions + Context + Question. """
    context = "\n\n---\n\n".join(
        f"[Source: {chunk.metadata.get('source','?')}, Page: {chunk.metadata.get('page_label','?')}]\n{chunk.page_content}"
        for chunk in chunks
    )
    prompt = f"""You are a helpful assistant answering questions based on the provided context.
Use only the information in the context. If the answer isn't in the context, say so.
Always cite the source and page when possible.

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def call_llm(prompt: str) -> str:
    """Send the prompt to OpenAI and return the answer text."""
    response = _llm.invoke(prompt)
    return response.content

def query_documents(question: str , k: int = 3):
    """Run the full RAG Query: retrieve -> build prompt -> call LLM."""
    chunks = retrieve_chunks(question,k=k)
    prompt = build_prompt(question,chunks)
    answer = call_llm(prompt)
    sources = [
        {
            "source": c.metadata.get("source"),
            "page": c.metadata.get("page_label"),
        }
        for c in chunks 
    ]
    return {"answer": answer , "sources": sources} 