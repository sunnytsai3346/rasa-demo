import os
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from sentence_transformers import SentenceTransformer
import faiss
import json
import requests

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query
EMBED_MODEL = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "DATA","FAISS_index")
CHUNK_SIZE = 300
TOP_K = 3
METADATA_STORE = "vector_meta.pkl"

class ActionRAGFallback(Action):
    def name(self):
        return "action_rag_fallback"

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))     
        with open(os.path.join(INDEX_PATH, METADATA_STORE), encoding="utf-8") as f:
            self.docs = json.load(f)   
        

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")

        # Step 1: Embed user input        
        query_vec = self.embedder.encode([user_input], convert_to_numpy=True)
        

        # Step 2: Search FAISS index
        D, I = self.index.search(query_vec, k=TOP_K)
        retrieved_texts = [self.docs[i]["text"] for i in I[0]]

        # Step 3: Compose context for LLM
        context = "\n\n".join(retrieved_texts)
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_input}
Answer:"""

        # Step 4: Call LLaMA via Ollama or your LLM provider
        res = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })

        answer = res.json()["response"]
         # Log query to CSV
        log_summary_query(user_input, "fallback_rag_llm", answer)
        dispatcher.utter_message(text=answer)
        return []
