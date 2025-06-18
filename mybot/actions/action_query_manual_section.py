import os
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher
from actions.knowledgebase import PDFKnowledgeBase
from actions.log.log_summary_query import log_summary_query
PDF_PATH = os.path.join(os.path.dirname(__file__), "PDF","manual.pdf")
knowledge_base = PDFKnowledgeBase(PDF_PATH,debug=True)


class ActionQueryManualSection(Action):
    def name(self):
        return "action_query_manual_section"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        query = tracker.latest_message.get("text")
        section_title, summary = knowledge_base.search_by_title(query)
        # Log query to CSV
        log_summary_query(query, section_title, summary)
        dispatcher.utter_message(text=f"**{section_title}**\n{summary}")
        return [SlotSet("section_title", section_title)]  
