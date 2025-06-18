import csv
import json
import logging
import os
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher
from langdetect import detect,DetectorFactory
from fuzzywuzzy import fuzz
from rapidfuzz.fuzz import partial_ratio

# Set up logger
log_file_path = os.path.join(os.path.dirname(__file__), "log","user_inputs.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# def detect_language(text: str) -> str:
#     # from langdetect import detect, DetectorFactory
#     # DetectorFactory.seed = 0

#     SUPPORTED_LANGS = {"en", "de", "es", "it","ru","fr","ko","ja","zh"}
    
#     try:
#         # script_lang = detect_script(text)
#         # if script_lang != "unknown":
#         #     return script_lang
#         # # Fall back to langid for European languages
#         # lang_code, score = langid.classify(text)
#         # if score < 0.85 or len(text) < 5:
#         #     return "en"
        
#         #return lang_code if lang_code in SUPPORTED_LANGS else "en"
#         if re.search(r'[english]', text.lower, re.IGNORECASE):
#             return "en"
#         elif re.search(r'[en]', text.lower, re.IGNORECASE):
#             return "en"
#         elif re.search(r'[german]', text.lower, re.IGNORECASE):
#             return "de"
#         elif re.search(r'[de]', text.lower, re.IGNORECASE):
#             return "de"
#     except:
#         return "en"
                