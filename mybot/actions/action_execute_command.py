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
        # The payload is expected in the format /command(value)
        match = re.match(r"/(.*)\((.*)\)", command_text)
        if match:
            command = match.group(1)
            value = match.group(2)
            # For now, just confirming the command that would be executed.
            dispatcher.utter_message(text=f"Executing: {command}({value})")
        else:
            dispatcher.utter_message(text=f"Sorry, I could not execute this command: {command_text}")
        return []
