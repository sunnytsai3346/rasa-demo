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


def detect_script(text: str) -> str:
    
    if re.search(r'[\u3040-\u30ff]', text):  # Japanese Hiragana/Katakana
        # print('detect_script',text,'ja')
        return "ja"
    elif re.search(r'[\uac00-\ud7af]', text):  # Korean Hangul
        # print('detect_script',text,'ko')
        return "ko"
    elif re.search(r'[\u4e00-\u9fff]', text):  # Chinese
        # print('detect_script',text,'zh')
        return "zh"
    # Russian Cyrillic
    elif re.search(r'[\u0400-\u04FF]', text):
        return 'ru'
    # Polish (has ł, ż, etc.)
    elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
        return 'pl'
    # German (has ü, ä, ö, ß)
    elif re.search(r'[äöüß]', text, re.IGNORECASE):
        return 'de'
    # Spanish (has ñ, á, é, í, ó, ú, ü)
    elif re.search(r'[áéíóúüñ]', text, re.IGNORECASE):
        return 'es'
    # French (é, è, ê, ç, etc.)
    elif re.search(r'[àâçéèêëîïôûùüÿ]', text, re.IGNORECASE):
        return 'fr'
     # Italian (à, è, é, ì, ò, ù)
    elif re.search(r'[àèéìòù]', text, re.IGNORECASE):
        return 'it'
    # Portuguese (ã, õ, ç, ê, etc.)
    elif re.search(r'[ãõâêîôûç]', text, re.IGNORECASE):
        return 'pt'
    print('detect_script',text,'unknown')
    return "unknown"        

def detect_language(text: str) -> str:
    # from langdetect import detect, DetectorFactory
    # DetectorFactory.seed = 0

    SUPPORTED_LANGS = {"en", "de", "es", "it","ru","fr","ko","ja","zh"}
    
    try:
        # script_lang = detect_script(text)
        # if script_lang != "unknown":
        #     return script_lang
        # # Fall back to langid for European languages
        # lang_code, score = langid.classify(text)
        # if score < 0.85 or len(text) < 5:
        #     return "en"
        
        #return lang_code if lang_code in SUPPORTED_LANGS else "en"
        if re.search(r'[english]', text.lower, re.IGNORECASE):
            return "en"
        elif re.search(r'[en]', text.lower, re.IGNORECASE):
            return "en"
        elif re.search(r'[german]', text.lower, re.IGNORECASE):
            return "de"
        elif re.search(r'[de]', text.lower, re.IGNORECASE):
            return "de"
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
            #sentences = [sent.text for sent in parsed.sents if user_input  in sent.text.lower()]            
            sentences = [sent.text for sent in parsed.sents]            
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
    

## ActionSearchKeyword ##
class ActionSearchKeyword(Action):
    def name(self) -> Text:
        return "action_search_keyword"
    def load_keywords(self, lang_code: str) -> Dict:
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
            "zh-cn": "ZH.json"
        }

        # default to English if language unsupported
        file_name = lang_map.get(lang_code, "EN.json")        
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
        
        user_input = tracker.latest_message.get("text", "").lower()        
        # Log the raw user input
        logger.info(f"User input: {user_input}")
        # Save user input to CSV
        self.save_to_csv(user_input)

        #lang_code = detect_language(user_input)
        lang_code = detect(user_input)
        print('215',lang_code)        
        
        keywords_data = self.load_keywords(lang_code)
        base_url = "http://192.168.230.169/"
        threshold = 75  # Fuzzy match threshold (0-100)

        matches = []

        for item in keywords_data:     
            #print(item)
            name = item["name"]
            url = item["url"]
            level = item.get("userLevel", 0)

            #score = fuzz.partial_ratio(user_message, name.lower())
            if len(name.strip())>0  : 
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
                    print(name,url)

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
            return [SlotSet("keyword", None)]
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