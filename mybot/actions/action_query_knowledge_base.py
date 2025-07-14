import os
import json
from typing import Any, Dict, List, Text

import faiss
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query

# --- Constants ---
EMBED_MODEL = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "DATA", "FAISS_index")
METADATA_FILE = "vector_meta.pkl"
TOP_K = 3


class ActionQueryKnowledgeBase(Action):
    """
    Answers user questions by combining document retrieval from a FAISS vector
    database with response generation from a Large Language Model (RAG).
    """

    def name(self) -> Text:
        """Returns the name of the action."""
        return "action_query_knowledge_base"

    def __init__(self, **kwargs):
        """Initializes the action, loading the embedding model and FAISS index."""
        super().__init__(**kwargs)
        try:
            self.embedder = SentenceTransformer(EMBED_MODEL)
            self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
            with open(os.path.join(INDEX_PATH, METADATA_FILE), "r", encoding="utf-8") as f:
                self.docs = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base: {e}")

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """
        Executes the RAG pipeline.

        1. Embeds the user's query.
        2. Searches the FAISS index for relevant document chunks.
        3. Constructs a prompt with the retrieved context.
        4. Calls the LLM to generate an answer.
        5. Dispatches the answer to the user.
        """
        user_input = tracker.latest_message.get("text")

        # 1. Embed user input
        query_vec = self.embedder.encode([user_input], convert_to_numpy=True)

        # 2. Search FAISS index
        _, indices = self.index.search(query_vec, k=TOP_K)
        retrieved_texts = [self.docs[i]["text"] for i in indices[0]]
        context = "\n\n".join(retrieved_texts)

        # 3. Compose prompt for LLM
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_input}
Answer:"""

        # 4. Call LLM and get response
        try:
            res = requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            res.raise_for_status()
            answer = res.json().get("response", "").strip()

            if not answer:
                answer = "I found some relevant information, but I couldn't generate a specific answer. You may want to review the retrieved documents."

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            answer = "I'm sorry, but I'm having trouble connecting to my knowledge source. Please try again later."

        # Log and dispatch the final answer
        log_summary_query(user_input, "rag_query", answer)
        dispatcher.utter_message(text=answer)

        return []