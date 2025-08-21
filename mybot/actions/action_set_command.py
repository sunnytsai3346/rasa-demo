from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


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
        """
        Presents buttons to the user to select a 'set' command.
        """
        buttons = [
            {"title": "Update projector start date to Wednesday", "payload": "setStartDayOfWeek(3)"},
            {"title": "Set 60 sec session timeout", "payload": "setScreenTimeout(3600000)"},
            {"title": "Set Temperature to Celsius", "payload": "setTemperatureUnits(0)"},
        ]

        dispatcher.utter_message(
            text="What setting would you like to change?",
            buttons=buttons
        )

        return []


