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
#BASE_URL ='http://192.168.230.169'



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
        """
        Checks for a status query using intelligent keyword matching and
        returns an answer if a high-confidence match is found.
        """
        stop_words = set(["what", "is", "are", "the", "tell", "me", "about"])
        query_words = set(re.findall(r'\w+', query.lower())) - stop_words

        if not query_words:
            return None, None

        best_match = None
        highest_score = 0.0

        for entry in self.status_data:
            name = entry.get("name", "")
            if not name:
                continue

            name_words = set(re.findall(r'\w+', name.lower()))
            intersection = query_words.intersection(name_words)
            union = query_words.union(name_words)
            score = len(intersection) / len(union) if union else 0.0

            if score > highest_score:
                highest_score = score
                best_match = entry

        # Only proceed if we have a reasonably strong match
        if highest_score > 0.5 and best_match:
            name = best_match.get("name")
            value = best_match.get("value")
            url = best_match.get("url", "")
            
            full_url = ""
            if url:
                if url.startswith('http://') or url.startswith('https://'):
                    full_url = url
                else:
                    full_url = f"{BASE_URL}{url}"
            
            topics = [f"{full_url} - {name} - {highest_score:.2f}"] if full_url else [f"{name} - {highest_score:.2f}"]
            
            prompt = f"You are a helpful assistant. Based on the following status, answer the user's question concisely.\n\nStatus:\n- {name}: {value}\n\nUser Query:\n{query}\n\nAnswer:"

            try:
                res = requests.post(
                    OLLAMA_URL,
                    json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                    timeout=30,
                )
                res.raise_for_status()
                answer = res.json().get("response", "Sorry, I couldn't generate an answer.")
                
                log_summary_query(query, f"{name}: {value} (url: {full_url})", answer, topics)
                return answer, topics
            except requests.exceptions.RequestException as e:
                print(f"Error calling LLM for status query: {e}")
                return "I'm sorry, but I'm having trouble retrieving that status.", []

        return None, None

    def _get_context_answer(self, query: str, topics: List[str], top_k: int = 3) -> List[str]:
        """
        Finds the most relevant context items based on keyword matching,
        ignoring common stop words.
        """
        # A simple set of stop words, can be expanded
        stop_words = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
            "have", "has", "had", "do", "does", "did", "what", "who", "when",
            "where", "why", "how", "which", "that", "this", "these", "those",
            "in", "on", "at", "for", "to", "from", "of", "with", "by", "file",
            "document", "page"
        ])

        # Tokenize and filter query
        query_words = set(re.findall(r'\w+', query.lower())) - stop_words

        if not query_words:
            return topics  # Cannot match on a query with only stop words

        matches = []
        for entry in self.context_data:
            original_name = entry.get("name") or ""
            value = entry.get("value") or ""
            url = entry.get("url", "")

            # Combine name and value for a fuller context
            context_text = f"{original_name.lower()} {value.lower()}"
            context_words = set(re.findall(r'\w+', context_text)) - stop_words

            if not context_words:
                continue

            # Calculate Jaccard similarity
            intersection = query_words.intersection(context_words)
            union = query_words.union(context_words)
            score = len(intersection) / len(union) if union else 0.0

            if score > 0.1:  # Only consider matches with some meaningful overlap
                matches.append({"score": score, "url": url, "name": original_name})

        # Sort matches by score (descending) and take the top_k
        sorted_matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:top_k]

        for match in sorted_matches:
            url = match.get("url")  # Safely get the url
            if url:  # Proceed only if url is not None and not an empty string
                if url.startswith('http://') or url.startswith('https://'):
                    full_url = url
                else:
                    full_url = f"{BASE_URL}{url}"
                topics.append(f"{full_url} - {match['name']} - {match['score']:.2f}")
            else:
                # If no URL, format without it.
                topics.append(f"{match['name']} - {match['score']:.2f}")

        return topics

    def _get_combined_answer(self, query: str) -> (str, List[str], bool):
        """
        Gets an answer by combining results from RAG (Top 1) and
        context search (Top 3).
        """
        # 1. Get Top 1 from RAG
        rag_context = ""
        rag_sources = []
        rag_score_is_low = True # Default to true

        query_vec = self.embedder.encode([f"query: {query}"], convert_to_numpy=True)
        scores, indices = self.index.search(query_vec, k=1)

        if scores[0][0] >= SCORE_THRESHOLD:
            rag_score_is_low = False
            score, doc_index = scores[0][0], indices[0][0]
            chunk = self.docs[doc_index]
            rag_context = f"[From Knowledge Base - Score: {score:.3f}] [{chunk['meta']['file']}]\n{chunk['text']}"
            src = chunk.get("meta", {}).get("file")
            if src:
                rag_sources.append(f"{src} (score: {score:.2f})")

        # 2. Get Top 3 from Context
        context_topics = self._get_context_answer(query, [], top_k=3)
        
        # 3. Combine and check if we have anything
        if not rag_context and not context_topics:
            log_summary_query(query, "", "No good answer found", [])
            return None, [], True

        # 4. Build Prompt and Call LLM
        combined_context = rag_context
        if context_topics:
            # Each item in context_topics is already a full string
            context_str = "\n- ".join(context_topics)
            combined_context += f"\n\n[From Related Topics]:\n- {context_str}"

        all_sources = rag_sources + context_topics
        
        prompt = f"You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{combined_context}\n\nQuestion:\n{query}\n\nAnswer:"

        try:
            res = requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            res.raise_for_status()
            answer = res.json().get("response", "").strip() or "I found some relevant information, but I couldn't generate a specific answer."

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM for combined query: {e}")
            answer = "I'm sorry, but I'm having trouble connecting to my knowledge source."
            all_sources = []
        
        log_summary_query(query, combined_context, answer, all_sources)
        return answer, all_sources, rag_score_is_low

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

        if answer is None:
            answer, topics, rag_score_is_low = self._get_combined_answer(query)

        return [
            SlotSet("kb_answer", answer),
            SlotSet("related_topics", topics),
            SlotSet("rag_score_is_low", rag_score_is_low),
        ]