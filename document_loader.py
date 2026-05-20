import fitz  # PyMuPDF
import os

def load_pdf(file_path):
    """Read a PDF file and return all text as a single string."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def load_and_chunk_pdf(file_path):
    """Main function: load PDF and return chunks."""
    text = load_pdf(file_path)
    chunks = split_into_chunks(text)
    return chunks