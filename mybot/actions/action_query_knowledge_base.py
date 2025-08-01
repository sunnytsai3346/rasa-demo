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

from actions.actions import BASE_URL, LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query

# --- Constants ---
EMBED_MODEL = "intfloat/multilingual-e5-large"  #"intfloat/e5-large-v2"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_PATH, "FAISS_index")
METADATA_FILE = "vector_meta.pkl"
STATUS_FILE = "status_data.json"
CONTEXT_FILE = "en_ts_processed.json"
TOP_K = 3
SCORE_THRESHOLD = 0.5  # Confidence threshold for RAG results
BASE_URL ='http://192.168.230.169'



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
            with open(os.path.join(DATA_PATH, CONTEXT_FILE), "r", encoding="utf-8") as f:
                self.context_data = json.load(f)    
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base or status data: {e}")
    def _get_response_intro(self, intent: str) -> str:
        """Tone-aware prefix based on intent"""
        if intent == "ask_politely":
            return "Thanks for your question!  Here's what I found:\n\n"
        elif intent == "ask_direct":
            return "Sure! Here's the info you asked for:\n\n"
        else:
            return "Here’s what I found:\n\n"    

    def _get_status_answer(self, query: str) -> (str, List[str]):
        """Checks for a status query and returns the answer and topics if found."""
        best_score = 0.0
        for entry in self.status_data:
            name = entry.get("name", "")
            if name and _word_in_query(name, query):
                value = entry.get("value")
                url = entry.get("url", "")
                if url:
                    url = f"{BASE_URL}{url}"  # exact match
                    best_score = 1.0
                    topics = [f"{BASE_URL}{url} - {name} - {best_score}"]                       
                    topics = self._get_context_answer(query,topics)
                    
                prompt = f"You are a helpful assistant. Based on the following status, relevant, answer the user's question concisely.\n\nStatus:\n- {name}: {value}\n\nRelevant:\n-{topics}\n\nUser Query:\n{query}\n\nAnswer:"

                try:
                    res = requests.post(
                        OLLAMA_URL,
                        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                        timeout=30,
                    )
                    res.raise_for_status()
                    answer = res.json().get("response", "Sorry, I couldn't generate an answer.")                    
                    
                    log_summary_query(query, f"{name}: {value} (url: {url})", answer,[f"{url} - {name}"])
                    return answer, topics
                except requests.exceptions.RequestException as e:
                    print(f"Error calling LLM for status query: {e}")
                    return "I'm sorry, but I'm having trouble retrieving that status.", []
        return None, None
    
    def _get_context_answer(self, query: str, topics: List[str]) -> List[str]:
        matches = []
        query_lower = query.lower()

        for entry in self.context_data:
            original_name = entry.get("name") or ""
            name_lower = original_name.lower()
            value_lower = (entry.get("value") or "").lower()
            url = entry.get("url", "")
            score = 0.0

            if name_lower in query_lower or value_lower in query_lower:
                score = 0.9
            elif any(word in query_lower for word in name_lower.split()):
                score = 0.6

            if score > 0:
                matches.append({"score": score, "url": url, "name": original_name})

        # Sort matches by score (descending) and take the top 4
        sorted_matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:4]

        for match in sorted_matches:
            topics.append(f"{BASE_URL}{match['url']} - {match['name']} - {match['score']}")

        return topics


    def _get_rag_answer(self, query: str) -> (str, List[str], bool):
        """Gets an answer using the RAG pipeline."""
        query_vec = self.embedder.encode([f"query: {query}"], convert_to_numpy=True)
        scores, indices = self.index.search(query_vec, k=TOP_K)

        if scores[0][0] < SCORE_THRESHOLD:
            log_summary_query(query, "", "No good answer found", [])
            return None, [], True  # answer, topics, rag_score_is_low

        context_parts = []
        related_sources = []
        seen_sources = set()
        matched = [(score, self.docs[i]) for score, i in zip(scores[0], indices[0])]
        context = "\n".join([f"[Score: {score:.3f}] [{chunk['meta']['file']}]\n{chunk['text']}" for score, chunk in matched])
        for score, chunk in matched:
            src = chunk.get("meta", {}).get("file")
            if src and src not in seen_sources:
                seen_sources.add(src)
                related_sources.append(f"{src} (score: {score:.2f})")
                
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

        
        log_summary_query(query, context, answer,related_sources)
        return answer, related_sources, False


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
        intent = tracker.latest_message.get("intent", {}).get("name", "")
        intro = self._get_response_intro(intent)

        answer, topics = self._get_status_answer(query)
        rag_score_is_low = False

        if answer is  None:
            answer, topics, rag_score_is_low = self._get_rag_answer(query)

        return [
            SlotSet("kb_answer", answer),
            SlotSet("related_topics", topics),
            SlotSet("rag_score_is_low", rag_score_is_low),
        ]