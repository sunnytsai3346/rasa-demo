import csv
import json
import re
import sys
import os

from actions.logger_util import log_summary_query
from actions.emotion_model import detect_emotion
from actions.knowledgebase import PDFKnowledgeBase
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)
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
import spacy
from fuzzywuzzy import fuzz
from rapidfuzz.fuzz import partial_ratio

import fitz  # PyMuPDF
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List, Text
from langdetect import detect,DetectorFactory

#from actions import config
USER_INTENT_OUT_OF_SCOPE = "out_of_scope"
en_spacy = spacy.load("en_core_web_md")

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
PDF_PATH = os.path.join(os.path.dirname(__file__), "PDF\manual.pdf")

knowledge_base = PDFKnowledgeBase(PDF_PATH, debug=True)




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
    
#ActionParsingUserGuide (action_parsing_userguide)
# run - parse manual.pdf
# no use now (intent :request_pdf_parsing)
# use ActionQueryManual 


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
    

## ActionSearchKeyword  action_search_keyword##
# intent - 
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



# ActionQueryManual (action_query_manual)
# run -query knowledge_base.search
# Let action_query_manual only handle setting slots
class ActionQueryManual(Action):
    
    def name(self):
        return "action_query_manual"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        query = tracker.latest_message.get("text")        
        #
        #{
        #"title": ...,
        #"summary": ...,
        #"content": ...,
        #"related": ...
        #}
        result = knowledge_base.search(query) 
        print('result:',result)
        if result is None or len(result) != 2:
            section_title, summary = None, None
        else:
            section_title = result[0]["title"]
            summary = result[0]["summary"]   
            related_topics = result[0]["related"]   
        
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
        print ('Related',related_topics)

        
        #dispatcher.utter_message(text=answer)
        return [SlotSet("kb_answer", json_answer),
                SlotSet("related_topics", related_topics)]

def clean_title(text):
    # Take first line, truncate long lines
    return text.split("\n")[0][:40] + "..." if len(text) > 40 else text.split("\n")[0]     
      
#  ActionQueryManualSection(action_query_manual_section):
# run -query knowledge_base.search_by_title
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


#ActionAcknowledgeEmotion(Action)
#Empathetic Response Templates
class ActionAcknowledgeEmotion(Action):
    def name(self):
        return "action_acknowledge_emotion"

    def run(self, dispatcher, 
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:
            
        emotion = tracker.latest_message.get("emotion", "neutral")
        
        #emotion, score = detect_emotion(tracker.latest_message)

        # if emotion == "anger":
        #     dispatcher.utter_message(response="utter_acknowledge_frustration")
        # elif emotion == "joy":
        #     dispatcher.utter_message(response="utter_acknowledge_happiness")
        # else:
        #     dispatcher.utter_message(response="utter_acknowledge_neutral")
        
        return [SlotSet("emotion", emotion)]


#ActionAnswerWithIntro (action_answer_with_intro: kb_answer + human like answer)
#Let action_answer_with_intro handle all utterances
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