# Install requirements first:
# pip install sentence-transformers qdrant-client PyMuPDF ollama

import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

PDF_DIR = "./pdfs"
EMBED_MODEL = "intfloat/e5-large-v2"
QDRANT_COLLECTION = "company_docs"

# 1. Load and chunk PDFs
def load_pdfs_to_chunks(pdf_dir):
    chunks = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_dir, file))
            for page in doc:
                text = page.get_text()
                if len(text.strip()) > 100:
                    chunks.append({"text": text, "meta": {"file": file}})
    return chunks

# 2. Embed chunks
def embed_chunks(chunks, model):
    texts = [f"query: {chunk['text']}" for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

# 3. Save to Qdrant
def save_to_qdrant(chunks, embeddings):
    client = QdrantClient("localhost", port=6333)
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload=chunk
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

# 4. Custom action in Rasa (actions/actions.py)
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionQueryDocs(Action):
    def name(self):
        return "action_query_docs"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        query = tracker.latest_message.get("text")

        # Embed query
        model = SentenceTransformer(EMBED_MODEL)
        query_emb = model.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Search Qdrant
        client = QdrantClient("localhost", port=6333)
        hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=query_emb, limit=3)
        retrieved_text = "\n".join([hit.payload['text'] for hit in hits])

        # Call LLaMA3 via Ollama
        prompt = f"""
You are a helpful assistant. Based on the following context, answer the question.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        answer = response.json()["response"].strip()
        dispatcher.utter_message(answer)
        return []

# ---- To prepare the DB (run once):
if __name__ == "__main__":
    print("Loading and embedding PDFs...")
    model = SentenceTransformer(EMBED_MODEL)
    chunks = load_pdfs_to_chunks(PDF_DIR)
    embeddings = embed_chunks(chunks, model)
    save_to_qdrant(chunks, embeddings)
    print("Saved to Qdrant.")
