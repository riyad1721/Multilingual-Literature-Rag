from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


app = FastAPI()

# Paths
faiss_dir = "faiss_index"
index_path = os.path.join(faiss_dir, "index.faiss")
chunks_path = os.path.join(faiss_dir, "chunks.pkl")
model_name = "intfloat/multilingual-e5-large"

# Load FAISS index and chunks
index = faiss.read_index(index_path)
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer(model_name)

# Request Schema
class QueryRequest(BaseModel):
    query: str

def get_relevant_chunks(query, top_k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

def generate_answer_ollama(prompt, model="mistral", host="http://localhost:11434"):
    endpoint = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # Lower temperature for more focused answers
            "top_p": 0.85,
            "repeat_penalty": 1.1,
            "num_ctx": 3072  # Increased context window
        }
    }
    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

def format_prompt(context, question):
    return f"""আপনি একজন বুদ্ধিমান সহকারী। নিচের প্রসঙ্গ পড়ে প্রশ্নের উত্তর দিন।
                প্রসঙ্গ:
                {context}
                
                প্রশ্ন:
                {question}
                
                উত্তর:"""

def evaluate_rag(query, context_chunks, answer):
    context_text = " ".join(context_chunks)
    query_embedding = embedder.encode([query])
    context_embedding = embedder.encode([context_text])
    answer_embedding = embedder.encode([answer])
    relevance = cosine_similarity(query_embedding, context_embedding)[0][0]
    groundedness = cosine_similarity(answer_embedding, context_embedding)[0][0]
    return relevance, groundedness

# API Endpoint
@app.post("/ask")
async def ask_question(req: QueryRequest):
    query = req.query
    relevant_chunks = get_relevant_chunks(query)
    context = "\n\n".join(relevant_chunks)
    prompt = format_prompt(context, query)
    answer = generate_answer_ollama(prompt)
    relevance, groundedness = evaluate_rag(query, relevant_chunks, answer)
    response = {
        "question": str(query),
        "answer": str(answer),
        "relevance_score": round(float(relevance), 4),
        "groundedness_score": round(float(groundedness), 4)
    }
    return response
