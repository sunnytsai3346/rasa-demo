import csv
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
# -*- coding: utf-8 -*-
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Text, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
)

#from actions import config
import spacy
en_spacy = spacy.load("en_core_web_md")

# from actions.api import community_events
# from actions.api.algolia import AlgoliaAPI
# from actions.api.discourse import DiscourseAPI
# from actions.api.gdrive_service import GDriveService
# from actions.api.mailchimp import MailChimpAPI
# from actions.api.rasaxapi import RasaXAPI

USER_INTENT_OUT_OF_SCOPE = "out_of_scope"

# Set up logger
log_file_path = os.path.join(os.path.dirname(__file__), "user_inputs.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INTENT_DESCRIPTION_MAPPING_PATH = "actions/intent_description_mapping.csv"
## ActionSearchKeyword ##

# actions/action_parse_pdf.py

from fuzzywuzzy import fuzz
from rapidfuzz.fuzz import partial_ratio

import fitz  # PyMuPDF
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List, Text
from langdetect import detect,DetectorFactory
DetectorFactory.seed = 0

def safe_detect(text: str)->str :        
        #fallback to English 
        if len(text) <5 or text.isascii():
            print('safe_detect',text,"en")
            return "en"
        try:
            print('safe_detect',text,detect(text))
            return detect(text)
        except:
            return "en"

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
            sentences = [sent.text for sent in parsed.sents if user_input  in sent.text.lower()]            
            response = "\n".join(sentences[:5]) if sentences else "No relevant info found in the PDF."
            dispatcher.utter_message(text=response)
            # Sample: Basic parsing — you can add NLP/keyword extraction here
            # if "installation" in content.lower():
            #     dispatcher.utter_message(text="The PDF includes installation instructions.")
            # else:
            #     dispatcher.utter_message(text="Couldn't find installation info in the PDF.")
        except Exception as e:
            dispatcher.utter_message(text=f"Error reading PDF: {str(e)}")

        return []
    
# Ensure consistent language detection
DetectorFactory.seed = 0
## ActionSearchKeyword ##
class ActionSearchKeyword(Action):
    def name(self) -> Text:
        return "action_search_keyword"
    def load_keywords(self, file_name: str) -> Dict:
        
  
        #keywords subfolder
        file_path = os.path.join(os.path.dirname(__file__), "keywords", file_name)
        print(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading keyword file: {e}")
            return {}

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        lang_map = {
            "en": "EN.json",
            "de": "DE.json",
            "ja": "JA.json",
            "ko": "KO.json",
            "es": "ES.json",
            "fr": "FR.json",
            "it": "IT.json",
            "pl": "PL.json",
            "pt": "PT.json",
            "ru": "RU.json",
            "zh": "ZH.json"
        }
        
        language = tracker.get_slot("language")
        keyword = tracker.get_slot("keyword")
        
        # Detect language if not set
        if not language:
            try:
                language = detect(user_input)
                dispatcher.utter_message(f"Detected language: {language}")
            except Exception as e:
                dispatcher.utter_message("Could not detect language. Please specify (e.g., en, zh).")
                return [SlotSet("language", None)]       

       # Validate inputs
        if not keyword:
            dispatcher.utter_message("Please provide a keyword to search for.")
            return [SlotSet("language", language)]
        if not language:
            dispatcher.utter_message("Please specify a language (e.g., en, fr, zh).")
            return [SlotSet("language", None)]
        dispatcher.utter_message(f"Detected language: {language}, keyword:{keyword}")
        user_input = tracker.latest_message.get("text", "").lower()        
        
        # Log the raw user input
        logger.info(f"User input: {user_input}")
        # Save user input to CSV
        self.save_to_csv(user_input)

        if language not in lang_map:
            dispatcher.utter_message(f"Sorry, the language '{language}' is not supported. Choose from: {', '.join(lang_map.keys())}")
            return [SlotSet("language", None)]
            
        # default to English if language unsupported
        file_name = lang_map.get(language, "EN.json")      

        keywords_data = self.load_keywords(file_name)
        base_url = "http://192.168.230.169/"
        threshold = 75  # Fuzzy match threshold (0-100)

        matches = []

        for item in keywords_data:     
            #print(item)
            name = item["name"]
            url = item["url"]
            level = item.get("userLevel", 0)

            #score = fuzz.partial_ratio(user_message, name.lower())
            if name.lower() !='none':
                if user_input.lower() in name.lower() or fuzz.partial_ratio(user_input.lower(), name.lower()) > threshold:
                    if url.startswith("/#/"):
                        url = url.replace("/#/", base_url)
                    elif url.startswith("#/"):
                        url = url.replace("#/", base_url)

                    matches.append({
                        "name": name,
                        "url": url,
                        "level": int(level) if str(level).isdigit() else 0,  # safely cast to int
                        "score": int(fuzz.partial_ratio(user_input.lower(), name.lower()) )  # fuzz returns int, but cast anyway just in case
                    })

        if matches:
            # Sort by userLevel descending, then by fuzzy score
            #matches.sort(key=lambda x: (-x["level"], -x["score"]))
            matches.sort(key=lambda x: (-x["score"]))
            top_matches = matches[:5]  # Show top 5 results

            #response_lines = [f"- [{m['name']}]({m['url']}) (level {m['level']}, match {m['score']}%)"
            response_lines = [f"- [{m['name']}]({m['url']}) ( match {m['score']}%)"
                              for m in top_matches]

            dispatcher.utter_message(text="Here are the most relevant matches:\n" + "\n".join(response_lines))
        else:
            dispatcher.utter_message(response="utter_no_result")

        return []


    def save_to_csv(self, user_input: Text):
        file_path = os.path.join(os.path.dirname(__file__), "nlu_user_inputs.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # If file is new, write header
            if not file_exists:
                writer.writerow(["text", "intent", "entities"])

            # Append user input with placeholders for later labeling
            writer.writerow([user_input, "", ""])

class ActionResetSlots(Action):
    def name(self):
        return "action_reset_slots"

    def run(self, dispatcher, tracker, domain):
        print('ActionResetSlots')
        return [SlotSet("keyword", None), SlotSet("language", None)]            