# 4. Custom action in Rasa (actions/actions.py)
import json
import os
from typing import Any, Dict, List, Text
import faiss
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
from sentence_transformers import SentenceTransformer
from rasa_sdk.events import SlotSet


EMBED_MODEL = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "DATA","FAISS_index")
CHUNK_SIZE = 300
TOP_K = 3

class ActionQueryDocs(Action):
    def name(self):
        return "action_query_docs"
    
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
        with open(os.path.join(INDEX_PATH, "metadata.json"), encoding="utf-8") as f:
            self.chunks = json.load(f)

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        

        query = tracker.latest_message.get("text")

        # Embed query
        query_vec = self.model.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, TOP_K)
        matched = [self.chunks[i] for i in indices[0]]
        context = "\n".join([f"[{c['meta']['file']}]\n{c['text']}" for c in matched])

        # Search Faiss Index
        prompt = f"""
You are a helpful assistant. Based on the following context, answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })

        answer = response.json().get("response", "Sorry, I couldn't generate an answer.")

       
        # Return to follow-up action
        return [
            SlotSet("kb_answer", answer),
            SlotSet("related_topics", [c["source"] for c in matched if "source" in c])
        ]

        