# import csv
# import json
# import re
# import sys
# import os

# from actions.emotion_model import detect_emotion
# from actions.knowledgebase import PDFKnowledgeBase
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# # print(sys.path)
# # -*- coding: utf-8 -*-
# import logging
# import json
# from datetime import datetime
# from typing import Any, Dict, List, Text, Optional

# from rasa_sdk import Action, Tracker
# from rasa_sdk.types import DomainDict
# from rasa_sdk.forms import FormValidationAction
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import (
#     SlotSet,
#     UserUtteranceReverted,
#     ConversationPaused,
#     EventType,
# )
# import spacy
# from fuzzywuzzy import fuzz
# from rapidfuzz.fuzz import partial_ratio

# import fitz  # PyMuPDF
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from typing import Any, Dict, List, Text
# from langdetect import detect,DetectorFactory
# from actions.knowledgebase import PDFKnowledgeBase


# PDF_PATH = os.path.join(os.path.dirname(__file__), "PDF","manual.pdf")
# knowledge_base = PDFKnowledgeBase(PDF_PATH,debug=True)


OLLAMA_URL = "http://localhost:11434/api/generate"             
LLM_MODEL = "llama3"


