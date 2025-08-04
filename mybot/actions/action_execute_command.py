import re
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionExecuteCommand(Action):
    """
    Executes a command received from a button payload.
    It parses the command and its value, and dispatches a message.
    """
    def name(self) -> Text:
        return "action_execute_command"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        command_text = tracker.latest_message.get("text")
        dispatcher.utter_message(text=f"Executing: {command_text}")
        
        return []
