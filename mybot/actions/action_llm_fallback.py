import requests
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.actions import LLM_MODEL, OLLAMA_URL
from actions.log import log_summary_query


class ActionLLMFallback(Action):
    """
    A fallback action that uses a Large Language Model (LLM) to generate a response.
    """

    def name(self) -> Text:
        """Returns the name of the action."""
        return "action_llm_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """
        Executes the fallback action.

        It takes the user's latest message, sends it to an LLM to generate a
        response, logs the interaction, and dispatches the response to the user.
        Handles potential request errors gracefully.
        """
        user_input = tracker.latest_message.get("text")

        prompt = f"""You are a helpful assistant. Please provide a concise and natural answer to the following question:

User: {user_input}
Assistant:"""

        try:
            res = requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=30,  # Add a timeout for the request
            )
            res.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            reply = res.json().get("response", "").strip()

            if not reply:
                reply = "I'm sorry, I couldn't generate a response. Please try rephrasing."

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            reply = "I'm sorry, but I'm having trouble connecting to my knowledge source. Please try again later."

        # Log the query and response
        log_summary_query(user_input, "general fallback", reply)

        dispatcher.utter_message(text=reply)
        return []
