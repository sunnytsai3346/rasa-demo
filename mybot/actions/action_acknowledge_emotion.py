import os
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
from rasa_sdk.executor import CollectingDispatcher

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