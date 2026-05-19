class DocuMindException(Exception):
    """Base Class for all DocuMind-specific exceptions."""
    status_code = 500 
    error_code = "internal_error"
    message = "An internal error occurred."

class LLMUnavailableException(DocuMindException):
    status_code = 503
    error_code = "llm_unavailable"
    message = "The language model is temporarily unavailable. Please Try Again"

class RetrievalException(DocuMindException):
    status_code = 500
    error_code = "retrieval_failed"
    message = "Failed to retrieve relevant documents."

class IngestionException(DocuMindException):
    status_code = 500
    error_code = "ingestion_failed"
    message = "Failed to ingest the document."

class EmptyDocumentError(IngestionException):
    """Raised When a pdf Contains no extractable text."""
    status_code = 400
    error_code = "no_extractable_text"
    message = "No extractable text found. The PDF may be a scanned or image-only document "
      