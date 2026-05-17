from app.services.ingestion_service import compute_file_hash , load_pdf , split_into_chunks 
import pytest 

def test_hash_of_same_file_is_always_the_same():
    """If we hash test.pdf twice , both hashes must match."""
    hash_first_time = compute_file_hash("test.pdf")
    hash_second_time = compute_file_hash("test.pdf")
    assert hash_first_time == hash_second_time 

def test_hash_is_16_charecter_string():
    """The Hash should be always 16-char string"""
    result = compute_file_hash("test.pdf")
    assert len(result) == 16 
    assert isinstance(result,str)

def test_hashing_missing_file_raise_filenotfounderror():
    """Calling compute_file_hash on a missing file must raise FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        compute_file_hash("this_file_does_not_exist.pdf")

def test_load_pdf_returns_list_of_documents():
    """load_pdf should return a non-empty list of documents"""
    documents = load_pdf("test.pdf")
    assert isinstance(documents,list)
    assert len(documents) > 0 

def test_chunks_never_exceed_chunk_size_limit():
    """No chunk should be larger than the configured 800 chars"""
    documents = load_pdf("test.pdf")
    chunks = split_into_chunks(documents)
    for chunk in chunks:
        assert len(chunk.page_content) <= 800

def test_split_chunks_return_empty_for_empty_input():
    """An empty list of documents should produce zero chunks."""
    chunks = split_into_chunks([])
    assert chunks == [] 


