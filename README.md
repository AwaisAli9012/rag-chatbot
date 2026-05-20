# 📄 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own PDF documents using LLMs.

## 🧠 How It Works

1. Upload any PDF document
2. The app splits it into chunks and converts them to vectors using sentence transformers
3. When you ask a question, it finds the most relevant chunks using FAISS vector search
4. The relevant chunks are sent to LLaMA 3.1 via Groq API to generate a precise answer

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web interface |
| sentence-transformers | Text embeddings |
| FAISS | Vector similarity search |
| PyMuPDF | PDF text extraction |
| Groq API (LLaMA 3.1) | LLM for answer generation |
| python-dotenv | Environment variable management |

## 📁 Project Structure

rag-chatbot/
├── app.py               # Streamlit web interface
├── rag_engine.py        # Core RAG logic (embeddings, search, LLM)
├── document_loader.py   # PDF loading and text chunking
├── requirements.txt     # Project dependencies
└── .env                 # API keys (not included in repo)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AwaisAli9012/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and activate conda environment
```bash
conda create -n rag-chatbot python=3.11
conda activate rag-chatbot
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

### 5. Run the app
```bash
streamlit run app.py
```

## 💡 Key Concepts

**RAG (Retrieval-Augmented Generation)** — Instead of relying solely on the LLM's training data, RAG retrieves relevant information from your own documents and provides it as context to the LLM, resulting in accurate and document-specific answers.

**Vector Embeddings** — Text chunks are converted into numerical vectors that capture semantic meaning, enabling similarity-based search rather than exact keyword matching.

**FAISS** — Facebook's efficient similarity search library that finds the most relevant document chunks for any given question.