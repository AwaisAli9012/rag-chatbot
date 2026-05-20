import streamlit as st
from document_loader import load_and_chunk_pdf
from rag_engine import create_vector_store, search_similar_chunks, get_answer
import tempfile
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 RAG Chatbot")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing your PDF..."):
        chunks = load_and_chunk_pdf(tmp_path)
        index, chunks = create_vector_store(chunks)
    
    st.success(f"PDF processed! {len(chunks)} chunks created.")
    
    question = st.text_input("Ask a question about your PDF:")
    
    if question:
        with st.spinner("Searching for answer..."):
            relevant_chunks = search_similar_chunks(question, index, chunks)
            answer = get_answer(question, relevant_chunks)
        
        st.subheader("Answer:")
        st.write(answer)
        
        with st.expander("See relevant chunks used"):
            for i, chunk in enumerate(relevant_chunks):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk)
                st.divider()
    
    os.unlink(tmp_path)