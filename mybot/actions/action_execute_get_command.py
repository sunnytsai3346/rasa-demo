from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionExecuteGetCommand(Action):
    """
    Executes a 'get' command received from a button payload.
    """
    def name(self) -> Text:
        return "action_execute_get_command"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        command = tracker.latest_message.get("text")
        
        # This is a placeholder. You would add your logic here to call the
        # corresponding API or function to get the actual value.
        if command == "getStartDayOfWeek":
            response_text = "getStartDayOfWeek()"
        elif command == "getScreenTimeout":
            response_text = "getScreenTimeout()"
        elif command == "drawGraph":
            response_text = "drawGraph('All')"
        else:
            response_text = f"Sorry, I don't know how to get the information for: {command}"
        print(command)    

        dispatcher.utter_message(text=response_text)
        return []
