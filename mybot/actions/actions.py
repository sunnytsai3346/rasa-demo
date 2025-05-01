import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
#from .action_search_keyword import ActionSearchKeyword
#from .action_parsing_userguide import ActionParsingUserGuide
# actions/action_parse_pdf.py

from fuzzywuzzy import fuzz
# print(fuzz.partial_ratio("3d", "3D Sync"))
from rapidfuzz.fuzz import partial_ratio

import fitz  # PyMuPDF
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List, Text

class ActionParsePdf(Action):
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
        pdf_path = "files/sample.pdf"  # Replace with dynamic path if needed
        print(pdf_path)
        try:
            content = self.extract_text_from_pdf(pdf_path)
            # Sample: Basic parsing — you can add NLP/keyword extraction here
            if "installation" in content.lower():
                dispatcher.utter_message(text="The PDF includes installation instructions.")
            else:
                dispatcher.utter_message(text="Couldn't find installation info in the PDF.")
        except Exception as e:
            dispatcher.utter_message(text=f"Error reading PDF: {str(e)}")

        return []
    


class ActionSearchKeyword(Action):
    def name(self) -> Text:
        return "action_search_keyword"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        file_path = os.path.join(os.path.dirname(__file__), "EN.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        user_message = tracker.latest_message.get("text", "").lower()
        base_url = "http://192.168.230.169/"
        threshold = 60  # Fuzzy match threshold (0-100)

        matches = []

        for item in data:
            name = item["name"]
            url = item["url"]
            level = item.get("userLevel", 0)

            #score = fuzz.partial_ratio(user_message, name.lower())
            score = partial_ratio(user_message, name.lower())
            if score >= threshold:
                if url.startswith("/#/"):
                    url = url.replace("/#/", base_url)
                elif url.startswith("#/"):
                    url = url.replace("#/", base_url)

                matches.append({
                    "name": name,
                    "url": url,
                    "level": int(level) if str(level).isdigit() else 0,  # safely cast to int
                    "score": int(score)  # fuzz returns int, but cast anyway just in case
                })

        if matches:
            # Sort by userLevel descending, then by fuzzy score
            matches.sort(key=lambda x: (-x["level"], -x["score"]))
            top_matches = matches[:10]  # Show top 10 results

            response_lines = [f"- [{m['name']}]({m['url']}) (level {m['level']}, match {m['score']}%)"
                              for m in top_matches]

            dispatcher.utter_message(text="Here are the most relevant matches:\n" + "\n".join(response_lines))
        else:
            dispatcher.utter_message(response="utter_no_result")

        return []
