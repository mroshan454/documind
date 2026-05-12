from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
import hashlib 

#Initialise the embedding model 
_embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-V2"
)

def load_pdf(file_path: str):
    """ Load a PDF file and return its pages as Document objects."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents 

def split_into_chunks(documents):
    """ Split a list of Documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150 
    )
    chunks = splitter.split_documents(documents)
    return chunks 

def embed_chunks(chunks):
    """Embed a list of chunks into 384-dim vectors."""
    texts = [chunk.page_content for chunk in chunks]
    embeddings = _embedding_model.embed_documents(texts)
    return embeddings 


CHROMA_DIR = "./chroma_db"

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents"""
    hasher = hashlib.sha256()
    with open(file_path,"rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
        return hasher.hexdigest()[:16]

def store_in_chromadb(chunks, file_hash:str):
    """Store Chunks in ChromaDB with deterministic IDs based on file hash."""
    ids = [f"{file_hash}-chunk-{i}" for i in range(len(chunks))]
    
    vector_store = Chroma(
        collection_name="documind",
        embedding_function=_embedding_model,
        persist_directory=CHROMA_DIR,
    )

    vector_store.add_documents(documents=chunks, ids=ids)
    return vector_store 

def ingest_document(file_path:str):
    """Run the full ingestion pipeline: hash -> load -> split -> store."""
    file_hash = compute_file_hash(file_path)
    documents = load_pdf(file_path)
    chunks = split_into_chunks(documents)
    store_in_chromadb(chunks,file_hash)
    return {
           "file_hash": file_hash,
           "pages": len(documents), 
           "chunks":len(chunks),
           }