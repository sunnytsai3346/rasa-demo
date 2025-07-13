import json
import os
import re
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
TOP_K = 10
METADATA_STORE = "vector_meta.pkl"

STATUS_JSON_PATH = os.path.join(os.path.dirname(__file__), "DATA")    

def load_status_dicts():
    with open(os.path.join(STATUS_JSON_PATH, "status_data.json"), encoding="utf-8") as f:
        return json.load(f)  # Must be a list of dicts like {"name": ..., "value": ..., "url": ...}

def word_in_query(word, query):
        return re.search(rf"\b{re.escape(word)}\b", query)

class ActionQueryStatusOrDocs(Action):
    def name(self):
        return "action_query_status_or_docs"

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
        with open(os.path.join(INDEX_PATH, METADATA_STORE), encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.status_data = load_status_dicts()
        print('status_data:',self.status_data)

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        query = tracker.latest_message.get("text", "").lower()
        print('query is ', query)

        # ✅ Match against status_data
        # matched_entry = next(
            
        #     (entry for entry in self.status_data if entry.get("name", "").lower() in query or str(entry.get("value", "")).lower() in query),
        #     None
        # )
        matched_entry = None
        for entry in self.status_data:
            name = entry.get("name", "").lower()
            value = str(entry.get("value", "")).lower()

            print(f"Checking entry: name='{name}', value='{value}' against query='{query}'")

            if word_in_query(name, query):
                matched_entry = entry
                print(f"Matched entry: {entry}")
                break

        if matched_entry:
            name = matched_entry.get("name")
            value = matched_entry.get("value")
            url = matched_entry.get("url", "")
            print('matched:',name,value,url)

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
                SlotSet("related_topics", [f"{url} - {name}"])
            ]

        # Fallback to semantic RAG
        if not matched_entry:
           print("No status entry matched, falling back to FAISS search.") 
       
        query_vec = self.model.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, TOP_K)
        matched = [(score, self.chunks[i]) for score, i in zip(scores[0], indices[0])]        
        context = "\n".join([f"[Score: {score:.3f}] [{chunk['meta']['file']}]\n{chunk['text']}" for score, chunk in matched])

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

        related_sources = list({
            chunk.get("meta", {}).get("source", "unknown")
            for _, chunk in matched
        })

        return [
            SlotSet("kb_answer", answer),
            SlotSet("related_topics", related_sources)
        ]
    
    
