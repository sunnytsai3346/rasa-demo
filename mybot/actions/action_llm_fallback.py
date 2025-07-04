from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, UserUttered
import requests

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query

class ActionLLMFallback(Action):
    def name(self):
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_input = tracker.latest_message.get("text")

        prompt = f"""
You are a helpful assistant. Answer this question naturally:

User: {user_input}
Assistant:"""

        res = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })
        reply = res.json()["response"].strip()
        # Log query to CSV
        log_summary_query(user_input, "general fallback", reply)
        dispatcher.utter_message(reply)
        return []