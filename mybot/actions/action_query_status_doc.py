import json
import os
from typing import Any, Dict, List, Text
import faiss
import fitz
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

STATUS_JSON_PATH = os.path.join(os.path.dirname(__file__), "DATA")    

def load_status_dicts():
    status_data = {}
    with open(os.path.join(STATUS_JSON_PATH, "status_data.json"), encoding="utf-8") as f:
    
        data = json.load(f)
        status_data.update(data)
    return status_data

class ActionQueryStatusOrDocs(Action):
    def name(self):
        return "action_query_status_or_docs"

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
        with open(os.path.join(INDEX_PATH, METADATA_STORE), encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.status_data = load_status_dicts()

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        query = tracker.latest_message.get("text", "").lower()
        matched_kv_url = next(((name, value,url) for name, value,url in self.status_data.items() if name.lower() in query or str(value).lower() in query ), None)

        if matched_kv_url:
            name, value,url = matched_kv_url
            prompt = f"""
You are a helpful assistant. Based on the following equipment status information, answer the user question.

Status:
- {name}: {value} (url: {url})

User Query:
{query}

Answer:
"""
            response = requests.post(OLLAMA_URL, json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            })
            answer = response.json().get("response", "Sorry, I couldn't generate an answer.")
            return [
                SlotSet("kb_answer", answer),
                SlotSet("related_topics", [name])
            ]

        # Else fallback to semantic search
        print('query:',query,',fallback to semantic search')
        query_vec = self.model.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, TOP_K)
        matched = [(scores[0][i], self.chunks[indices[0][i]]) for i in range(TOP_K)]

        context = "\n".join(
            [f"[Score: {score:.3f}] [{chunk['meta']['source']}]\n{chunk['text']}" for score, chunk in matched]
        )

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

        related_sources = list({c['meta'].get("source", "") for _, c in matched if "meta" in c})

        return [
            SlotSet("kb_answer", answer),
            SlotSet("related_topics", related_sources)
        ]
