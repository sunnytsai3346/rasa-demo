import os
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher

from actions.knowledgebase import PDFKnowledgeBase
from actions.log.log_summary_query import log_summary_query

PDF_PATH = os.path.join(os.path.dirname(__file__), "PDF","manual.pdf")
knowledge_base = PDFKnowledgeBase(PDF_PATH,debug=True)

class ActionQueryManual(Action):
    
    def name(self):
        return "action_query_manual"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        query = tracker.latest_message.get("text")        
        section_title, summary = knowledge_base.search(query)        
        
        if not isinstance(summary, str):
            summary = str(summary)  # or summary = summary[1] if it's a (title, summary) tuple
        summary = summary.replace("\n", " ").strip()
        
        # Log query to CSV
        log_summary_query(query, section_title, summary)

        # Split answer by sentence
        steps = [s.strip() for s in summary.split(". ") if s]
        steps = [s if s.endswith(".") else s + "." for s in steps]

        json_answer = {
            "type": "step_list",
            "title": "Here's how to proceed:",
            "steps": steps
        }
        
        # Use semantic similarity to get related topics
        related_topics = knowledge_base.get_related_topics(query)
        print ('331 Related',related_topics)

        
        #dispatcher.utter_message(text=answer)
        return [SlotSet("kb_answer", json_answer),
                SlotSet("related_topics", related_topics)]