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

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query


EMBED_MODEL = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "DATA","FAISS_index")
CHUNK_SIZE = 300
TOP_K = 3
METADATA_STORE = "vector_meta.pkl"

class ActionQueryDocs(Action):
    def name(self):
        return "action_query_docs"
    
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
        with open(os.path.join(INDEX_PATH, METADATA_STORE), encoding="utf-8") as f:
            self.chunks = json.load(f)

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        

        query = tracker.latest_message.get("text")

        # Embed query
        query_vec = self.model.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, TOP_K)        
        matched = [(score, self.chunks[i]) for score, i in zip(scores[0], indices[0])]        
        context = "\n".join([f"[Score: {score:.3f}] [{chunk['meta']['file']}]\n{chunk['text']}" for score, chunk in matched]
)

        # Search Faiss Index
        prompt = f"""
You are a helpful assistant. Based on the following context, answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })

        answer = response.json().get("response", "Sorry, I couldn't generate an answer.")

        # refine related_topics
        related_sources = []
        seen = set()
        for score, chunk in matched:
            src = chunk.get("meta", {}).get("file")
            if src and src not in seen:
                seen.add(src)
                related_sources.append(f"{src} (score: {score:.2f})")

        
         # Log query to CSV
        log_summary_query(query, context, answer,related_sources)
        
        # Return to follow-up action
        return [
            SlotSet("kb_answer", answer),
            #SlotSet("related_topics", [c["source"] for c in matched if "source" in c])
            #SlotSet("related_topics", list(set(c["source"] for c in matched if "source" in c)))
            SlotSet("related_topics", related_sources)
        ]

        