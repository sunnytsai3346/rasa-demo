import os
import re
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher
import requests

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
        print('31:',json_answer)

        if not json_answer:
            dispatcher.utter_message(text="Sorry, I couldn't find relevant section.")
            return []
        

        # # Call LLaMA3 via Ollama
        # prompt = f"""
        # You are a helpful assistant. Based on the following context, answer the question.
        # Context:
        # {json_answer}
        
        # Question:
        # ""
        
        # Answer:
        # """
        # response = requests.post("http://localhost:11434/api/generate", json={
        #     "model": "llama3",
        #     "prompt": prompt,
        #     "stream": False
        # })
        # answer = response.json()["response"].strip()
        dispatcher.utter_message(f"{prefix}\n\n{json_answer}")
        if related:
             #dispatcher.utter_message("This is also related:\n- " + "\n- ".join(related))
             dispatcher.utter_message("Reference:\n- " + "\n- ".join(related))
            
        # prefix = "I found something that may help you."
        # # ---------- 🔁 Handle structured response from LLaMA ----------
        # # Structured response
        # if isinstance(json_answer, dict) and json_answer.get("type") == "step_list":
        #      steps = json_answer.get("steps", [])
        #      title = json_answer.get("title", "")
        #      if steps and isinstance(steps, list):
        #         dispatcher.utter_message(json_message={
        #             "text": answer                    
        #             "isTyping": False
        #         })
        #      else:
        #         # Fallback if 'steps' malformed
        #         dispatcher.utter_message(text=f"{title}\n\n{json_answer}")

        # # ---------- 🔘 Related topic buttons ----------
        # if related and isinstance(related, list) and len(related) > 0:
        #     def clean_title(text):
        #         text = re.sub(r'^\d+\.\s*', '', text.strip())  # Remove numbered prefixes like "5. "
        #         return text.split("\n")[0][:40] + "..." if len(text) > 40 else text.split("\n")[0]

        #     seen_titles = set()
        #     buttons = []

        #     for topic in related:
        #         title = clean_title(topic)
        #         if title and title.lower() not in seen_titles:
        #             buttons.append({
        #                 "title": title,
        #                 "payload": f"Show me how to {title.lower()}"
        #             })
        #             seen_titles.add(title.lower())

        #     dispatcher.utter_message(
        #         text="Would you like to explore a related topic?",
        #         buttons=buttons
        #     )

        # return []