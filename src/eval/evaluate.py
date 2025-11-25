import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import json

def load_demo_model():
    model_path = "checkpoints/demo_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def evaluate(texts, labels):
    tokenizer, model = load_demo_model()
    preds = []

    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs)
            pred = torch.argmax(out.logits, dim=1).item()
            preds.append(pred)

    print("Classification Report:\n")
    print(classification_report(labels, preds))

    print("Confusion Matrix:\n")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    # sample testing texts
    sample_texts = [
        "I feel sad and tired",
        "Life is going great this week",
        "Nothing feels good anymore",
        "I'm doing really well today"
    ]

    sample_labels = [1, 0, 1, 0]   # pretend labels
    evaluate(sample_texts, sample_labels)
