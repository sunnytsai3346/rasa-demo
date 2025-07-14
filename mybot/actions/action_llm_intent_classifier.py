from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, UserUttered, FollowupAction
import requests
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query
class ActionLLMIntentClassifier(Action):
    def name(self):
        return "action_llm_intent_classifier"

    def run(self, dispatcher, tracker: Tracker, domain):
        user_input = tracker.latest_message.get("text")
        intent_list = [i["name"] for i in domain.get("intents", [])]

        prompt = f"""
You are a helpful assistant that classifies user messages into one of the following intents:
{intent_list}
User message: "{user_input}"
Only reply with the intent name. If unsure, reply with "unknown".
"""

        res = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })
        llm_intent = res.json()["response"].strip()
        

        if llm_intent in intent_list:
            # Log query to CSV
            log_summary_query(user_input, "intent calssifier", llm_intent)
            return [UserUttered(user_input, {"intent": {"name": llm_intent, "confidence": 1.0}})]
        else:
            # Decide fallback_type based on keywords or heuristics
            knowledge_keywords = ["manual", "reset", "setup", "guide", "feature","projector","calibration","keystone","IMB","laser",'light']
            
            if any(k in user_input.lower() for k in knowledge_keywords):
                # Log query to CSV
                log_summary_query(user_input, "intent calssifier", "action_llm_rag_fallback")           
                return [SlotSet("fallback_type", "knowledge"), FollowupAction("action_llm_rag_fallback")]
            else:
                log_summary_query(user_input, "intent calssifier", "action_llm_fallback")           
                return [SlotSet("fallback_type", "general"), FollowupAction("action_llm_fallback")]
