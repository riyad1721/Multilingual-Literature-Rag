# Multilingual-Literature-Rag

**Simple Multilingual Retrieval-Augmented Generation (RAG) System**

---

## 🤖 Bengali-English Conversational QA API (RAG-based)

This project is a RESTful API for a multilingual RAG (Retrieval-Augmented Generation) system capable of answering **Bangla and English queries** from a pre-embedded document corpus using:

- 📚 FAISS for vector similarity search  
- 🌐 SentenceTransformers for multilingual embeddings  
- 🔮 Ollama + Mistral for local LLM generation  
- ⚡ FastAPI for API service  

---

## 🚀 Quickstart

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/riyad1721/Multilingual-Literature-Rag.git
cd Multilingual-Literature-Rag
uv venv

# Windows
.\.venv\Scriptsctivate

# Linux/macOS
source .venv/bin/activate

uv add -r requirements.txt
```

---

## 🧠 Building the FAISS Vector Store (Embedding Pipeline)

This section guides you through preparing your Bangla PDF documents for semantic search by extracting text, generating embeddings, and creating a FAISS vector store.

### 📋 Prerequisites

* Active Python virtual environment

### 🚀 Step-by-Step Process

#### 1. Navigate to the Research Directory

```bash
cd research/
```

#### 2. Execute the Notebook

Open and run the `generate_embeddings.ipynb` notebook to:

* Extract text using OCR (for Bangla)
* Clean and chunk the text
* Generate multilingual sentence embeddings
* Build and save the FAISS index

---

## 📘 Bengali-English RAG QA API (FastAPI + FAISS + Ollama)

This is a simple, production-ready REST API that allows users to ask Bengali or English questions about a given corpus (already embedded with FAISS). The system uses:

* FAISS for vector search
* SentenceTransformer (`intfloat/multilingual-e5-large`) for multilingual embedding
* Ollama (e.g., Mistral) for answer generation
* FastAPI for the API framework

---

## ✨ Features

* ✅ Bengali & English question support
* ✅ Top-k document retrieval from pre-indexed FAISS vector store
* ✅ Response generation using locally running Ollama model
* ✅ Evaluation metrics:

  * 🔹 **Relevance**: similarity between query and retrieved context
  * 🔹 **Groundedness**: similarity between generated answer and context
* ✅ FastAPI interactive docs at `/docs`

---

## ▶️ Running the Server

Start the server using the following command:

```bash
uvicorn api:app --reload --port 8000
```

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---
