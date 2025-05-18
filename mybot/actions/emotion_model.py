from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Load model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for emotion classification
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Function to get top emotion
def detect_emotion(text):
    results = emotion_classifier(text)[0]  # Get list of emotion scores
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_results[0]
    return top_emotion['label'], top_emotion['score']

# Example usage
# text = "I'm really frustrated and tired of this."
# emotion, score = detect_emotion(text)
# print(f"Detected Emotion: {emotion} (Confidence: {score:.2f})")