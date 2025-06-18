
# actions/query_action.py
import os
import fitz
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher
import spacy
#from actions import config
USER_INTENT_OUT_OF_SCOPE = "out_of_scope"
en_spacy = spacy.load("en_core_web_sm")

class ActionParsingUserGuide(Action):
    def name(self) -> Text:
        return "action_parsing_userguide"

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Example: assuming the PDF is already uploaded and accessible
        file_name  = "manual.pdf"  # Todo : or a path from tracker slot
        doc = os.path.join(os.path.dirname(__file__), "pdf", file_name)
        print(doc)
        
        try:
            full_text = self.extract_text_from_pdf(doc)
            user_input = tracker.latest_message.get("text", "").lower()
            
            
            #full_text = " ".join(page.get_text() for page in doc)
            # Use spaCy for NLP parsing
            parsed = en_spacy(full_text)
            #sentences = [sent.text for sent in parsed.sents if user_input  in sent.text.lower()]            
            sentences = [sent.text for sent in parsed.sents]            
            response = "\n".join(sentences[:5]) if sentences else "No relevant info found in the PDF."
            dispatcher.utter_message(text=response)
            
        except Exception as e:
            dispatcher.utter_message(text=f"Error reading PDF: {str(e)}")

        return []
    