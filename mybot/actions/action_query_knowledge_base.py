import json
import os
import re
from typing import Any, Dict, List, Text

import faiss
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query

# --- Constants ---
EMBED_MODEL = "intfloat/e5-large-v2"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_PATH, "FAISS_index")
METADATA_FILE = "vector_meta.pkl"
STATUS_FILE = "status_data.json"
TOP_K = 3


def _word_in_query(word: str, query: str) -> bool:
    """Checks if a whole word is present in a query."""
    return re.search(rf"\b{re.escape(word)}\b", query, re.IGNORECASE) is not None


class ActionQueryKnowledgeBase(Action):
    """
    Finds an answer to a user's question by first checking for a direct
    status match, then falling back to a Retrieval-Augmented Generation (RAG)
    pipeline. The result is stored in slots for a subsequent action to use.
    """

    def name(self) -> Text:
        """Returns the name of the action."""
        return "action_query_knowledge_base"

    def __init__(self, **kwargs):
        """Initializes the action, loading all necessary models and data."""
        super().__init__(**kwargs)
        try:
            self.embedder = SentenceTransformer(EMBED_MODEL)
            self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
            with open(os.path.join(INDEX_PATH, METADATA_FILE), "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            with open(os.path.join(DATA_PATH, STATUS_FILE), "r", encoding="utf-8") as f:
                self.status_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base or status data: {e}")

    def _get_status_answer(self, query: str) -> (str, List[str]):
        """Checks for a status query and returns the answer and topics if found."""
        for entry in self.status_data:
            name = entry.get("name", "")
            if name and _word_in_query(name, query):
                value = entry.get("value")
                url = entry.get("url", "")
                prompt = f"You are a helpful assistant. Based on the following status, answer the user's question concisely.\n\nStatus:\n- {name}: {value}\n\nUser Query:\n{query}\n\nAnswer:"

                try:
                    res = requests.post(
                        OLLAMA_URL,
                        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                        timeout=30,
                    )
                    res.raise_for_status()
                    answer = res.json().get("response", "Sorry, I couldn't generate an answer.")
                    topics = [f"{url} - {name}"] if url else [name]
                    log_summary_query(query, f"STATUS: {name}", answer)
                    return answer, topics
                except requests.exceptions.RequestException as e:
                    print(f"Error calling LLM for status query: {e}")
                    return "I'm sorry, but I'm having trouble retrieving that status.", []
        return None, None

    def _get_rag_answer(self, query: str) -> (str, List[str]):
        """Gets an answer using the RAG pipeline."""
        query_vec = self.embedder.encode([f"query: {query}"], convert_to_numpy=True)
        scores, indices = self.index.search(query_vec, k=TOP_K)

        context_parts = []
        related_sources = []
        seen_sources = set()

        for score, i in zip(scores[0], indices[0]):
            chunk = self.docs[i]
            context_parts.append(chunk['text'])
            source = chunk.get("meta", {}).get("file", "Unknown source")
            if source not in seen_sources:
                related_sources.append(f"{source} (score: {score:.2f})")
                seen_sources.add(source)

        context = "\n\n".join(context_parts)
        prompt = f"You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

        try:
            res = requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            res.raise_for_status()
            answer = res.json().get("response", "").strip() or "I found some relevant information, but I couldn't generate a specific answer."
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM for RAG query: {e}")
            answer = "I'm sorry, but I'm having trouble connecting to my knowledge source."
            related_sources = []

        log_summary_query(query, "RAG_QUERY", answer)
        return answer, related_sources

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """
        Executes the action. It sets the 'kb_answer' and 'related_topics' slots.
        """
        query = tracker.latest_message.get("text", "")
        answer, topics = self._get_status_answer(query)

        if answer is None:
            answer, topics = self._get_rag_answer(query)

        return [
            SlotSet("kb_answer", answer),
            SlotSet("related_topics", topics),
        ]