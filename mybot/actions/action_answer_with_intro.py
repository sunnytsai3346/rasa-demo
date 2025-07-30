import re
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionAnswerWithIntro(Action):
    """
    An action that answers a question based on information from a knowledge base.
    It retrieves the answer and related topics from slots, formats them,
    and sends them to the user.
    """

    # --- Constants ---
    INTRO_MESSAGE = "To my best understanding, I found something that may help you."
    NO_ANSWER_MESSAGE = "Sorry, I couldn't find a relevant section."
    RELATED_TOPICS_HEADER = "Reference:"

    def name(self) -> Text:
        """Returns the name of the action."""
        return "action_answer_with_intro"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """
        Executes the action.

        Retrieves the knowledge base answer and related topics from the tracker's
        slots. If an answer is found, it's dispatched to the user with an
        introductory message. If related topics are present, they are also
        sent. If no answer is found, a corresponding message is sent.

        Args:
            dispatcher: The dispatcher to send messages to the user.
            tracker: The tracker for the current conversation state.
            domain: The domain of the assistant.

        Returns:
            An empty list of events.
        """
        kb_answer = tracker.get_slot("kb_answer")
        related_topics = tracker.get_slot("related_topics")

        if not kb_answer:
            dispatcher.utter_message(text=self.NO_ANSWER_MESSAGE)
            return []

        # Send the main answer
        # Format all URLs as clickable links
        answer_with_links = re.sub(
           r'(https?://[^\s]+)',
            r'<a href="\1" target="_blank">\1</a>',
            kb_answer
        )
       
        dispatcher.utter_message(text=f"{self.INTRO_MESSAGE}\n\n{answer_with_links}")

        # Send related topics if they exist
        if related_topics:
            def make_clickable(text: str) -> str:
                return re.sub(
                    r'(https?://[^\s]+)',
                    r'<a href="\1" target="_blank">\1</a>',
                    text
                )
            # Process each topic line, and format it with hyperlink (if any)
            formatted_items = []
            for topic in related_topics:
                topic_with_links = make_clickable(topic)
                formatted_items.append(f"- {topic_with_links}")
            
            # Join them into one string for display
            related_topics_formatted = "\n".join(formatted_items)
            
            dispatcher.utter_message(
                text=f"{self.RELATED_TOPICS_HEADER}\n- {related_topics_formatted}"
            )

        return []
