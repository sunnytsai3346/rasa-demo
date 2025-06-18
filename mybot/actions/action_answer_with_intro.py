import os
import re
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher

class ActionAnswerWithIntro(Action):
    def name(self) -> Text:
        return "action_answer_with_intro"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Simulate getting the answer from your knowledge base
        json_answer = tracker.get_slot("kb_answer")
        related = tracker.get_slot("related_topics")
        prefix = "I found something that may help you."

        if not json_answer:
            dispatcher.utter_message(text="Sorry, I couldn't find an answer.")
            return []
        

        # Determine emotion-based prefix
        emotion = tracker.get_slot("emotion")
        print('emotion:',emotion)
        if emotion == "sadness":
            prefix = "I hope you're doing okay. Here's the answer I found for you:\n"
        elif emotion == "joy":
            prefix = "Great energy! Here's what I found:\n"
        else:
            prefix = "I’ve found the answer for you. Here's how to proceed:\n"

        
        # Structured response
        if isinstance(json_answer, dict) and json_answer.get("type") == "step_list":
            steps = json_answer.get("steps", [])
            title = json_answer.get("title", "")
            dispatcher.utter_message(json_message={
                "type": "step_list",
                "title": prefix, 
                "steps": steps
            })

        else:
            full_text = f"{prefix}\n\n{str(json_answer)}"
            dispatcher.utter_message(text=full_text)

        # Handle related topic buttons
        if related and isinstance(related, list) and len(related) > 0:
            def clean_title(text):
                # Remove numeric prefixes like "1.", "2.", "5."
                text = re.sub(r'^\d+\.\s*', '', text.strip())  # Remove "5." at the start
                # Replace newlines and trim length
                return text.split("\n")[0][:40] + "..." if len(text) > 40 else text.split("\n")[0]

            seen_titles = set()
            buttons = []

            for topic in related:
                title = clean_title(topic)
                if title not in seen_titles:
                    buttons.append({
                        "title": title,
                        "payload": f"Show me how to {title.lower()}"
                    })
                    seen_titles.add(title)

            dispatcher.utter_message(
                text="Would you like to explore a related topic?",
                buttons=buttons
            )
    
        return []