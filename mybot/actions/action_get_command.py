from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionGetCommand(Action):
    def name(self) -> Text:
        return "action_get_command"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """
        Presents buttons to the user to select a 'get' command.
        """
        buttons = [
            {"title": "Get first day of week", "payload": "getStartDayOfWeek"},
            {"title": "Get session timeout setting", "payload": "getScreenTimeout"},
            {"title": "drawGraph", "payload": "drawGraph"},
        ]

        dispatcher.utter_message(
            text="What information would you like to retrieve?",
            buttons=buttons
        )

        return []

