from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionSetCommand(Action):
    def name(self) -> Text:
        return "action_set_command"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        command = tracker.latest_message.get("text")

        if command == "/setStartDayOfWeek":
            dispatcher.utter_message(text="setStartDayOfWeek(3)")
        elif command == "/setTemperatureUnits":
            dispatcher.utter_message(text="setTemperatureUnits(44)")
        elif command == "/setScreenTimeout":
            dispatcher.utter_message(text="setScreenTimeout(1800000)")    
        else:
            dispatcher.utter_message(text=f"Command '{command}' not recognized.")

        return []

