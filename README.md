# 📚 PDF Semantic Chunking and Embedding Pipeline

This project is a semantic document processing pipeline that extracts text from PDFs, semantically chunks the content, generates embeddings using HuggingFace, and stores the data in a Milvus vector database for fast semantic search. It includes caching mechanisms and flag-based embedding insertion control.

---

## 🧠 Key Features

- ✅ Extracts text from PDF files using `pdfplumber`
- ✅ Semantically chunks large text bodies using `SemanticChunker`
- ✅ Generates sentence embeddings with HuggingFace's `all-mpnet-base-v2`
- ✅ Stores and indexes embeddings in [Milvus](https://milvus.io/)
- ✅ Supports caching of processed chunks to avoid redundant work
- ✅ Supports various Milvus index types (`FLAT`, `IVF_FLAT`, `HNSW`)
- ✅ Optional BM25 and cosine similarity scoring (imported, not used yet)

---

## 📁 Project Structure

├── main.py (your script file)
├── cached_chunks.pkl (cached text chunks)
├── embeddings_inserted.flag (insertion flag file)
├── documents/ (your PDF files go here)
├── .env (environment variables)
└── README.md


---

## Install dependencies:

pip install -r requirements.txt

---

## Set up .env file:

# Example
GROQ_API_KEY=your_groq_api_key

---

## Start Milvus:

docker-compose up -d

---

## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/pdf-semantic-rag-pipeline.git
cd pdf-semantic-pipeline


