from typing import Any, Dict, List, Optional, Text, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class EmotionClassifier(GraphComponent):
    def __init__(self, classifier):
        self.classifier = classifier

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "EmotionClassifier":
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        return cls(classifier)

    def __init__(self, classifier):
        self.classifier = classifier

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            text = message.get("text")
            if text:
                emotions = self.classifier(text)[0]
                top_emotion = max(emotions, key=lambda x: x["score"])
                message.set("emotion", top_emotion["label"])
        return messages
    
    def process_training_data(self, training_data: TrainingData) -> TrainingData:
         self.process(training_data.training_examples)
         return training_data
