from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 

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

def store_in_chromadb(chunks):
    """Store Chunks + Embeddings in a persistent ChromaDB."""
    vector_store = Chroma.from_documents(
        documents= chunks,
        embedding=_embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name="documind"
    )
    return vector_store 

def ingest_document(file_path:str):
    """Run the full ingestion pipeline: load -> split -> embed -> store."""
    documents = load_pdf(file_path)
    chunks = split_into_chunks(documents)
    store_in_chromadb(chunks)
    return {"pages": len(documents), "chunks":len(chunks)}