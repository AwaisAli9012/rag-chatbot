import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(chunks):
    """Convert chunks to vectors and store in FAISS."""
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings, dtype='float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

def search_similar_chunks(query, index, chunks, top_k=3):
    """Find the most relevant chunks for a question."""
    query_vector = model.encode([query])
    query_vector = np.array(query_vector, dtype='float32')
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def get_answer(question, relevant_chunks):
    """Send question + relevant chunks to Groq and return answer."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
    
Context:
{context}

Question: {question}

Answer based only on the context provided above."""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content