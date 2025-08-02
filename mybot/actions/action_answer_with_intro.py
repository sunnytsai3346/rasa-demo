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
            formatted_items = []
            for topic in related_topics:
                # Expected format: "http://... - Topic Name - 0.67" or "Topic Name - 0.67"
                parts = topic.split(" - ")
                try:
                    # Case 1: URL is present
                    if len(parts) >= 3 and parts[0].startswith('http'):
                        url, name = parts[0], parts[1]
                        # Format as: <a href="url">Topic Name</a>
                        formatted_items.append(f'<a href="{url}" target="_blank">{name}</a>')
                    # Case 2: No URL, just a name and maybe a score
                    elif len(parts) >= 1:
                        name = parts[0]
                        # Handle cases where the topic might be from RAG (e.g., "filename.pdf (score: 0.85)")
                        if ' (score:' in name:
                            name = name.split(' (score:')[0]
                        formatted_items.append(name)
                    # Case 3: Fallback for any other unexpected format
                    else:
                        formatted_items.append(topic)
                except IndexError:
                    formatted_items.append(topic) # Fallback for safety

            # Join them into one string for display, using a line break
            related_topics_formatted = "<br>".join(formatted_items)

            dispatcher.utter_message(
                text=f"{self.RELATED_TOPICS_HEADER}<br>{related_topics_formatted}"
            )
            dispatcher.utter_message(response="utter_suggested_steps")

        return []
