from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionSetScreenTimeout(Action):
    def name(self) -> Text:
        return "action_set_screen_timeout"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="setStartDayOfWeek(2)") #test TUESDAY
        return []
