import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 1. Load Model & Tokenizer
# -----------------------------
MODEL_DIR = "checkpoints/demo_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# -----------------------------
# 2. Small Manual Test Set
# (You can replace these later)
# -----------------------------
texts = [
    "I feel tired and hopeless",
    "Life is meaningless",
    "I am okay today",
    "Everything is going fine"
]

y_true = [1, 1, 0, 0]   # Expected labels for test samples

# -----------------------------
# 3. Run model predictions
# -----------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return torch.argmax(probs).item()

y_pred = [predict(text) for text in texts]

# -----------------------------
# 4. Print Classification Report
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(
    y_true, 
    y_pred, 
    digits=2,
    zero_division=0     # <-- fixes the precision warning
))

# -----------------------------
# 5. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n")
print(cm)

# -----------------------------
# 6. Save Confusion Matrix Plot
# -----------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# -----------------------------
# 7. Save Bar Plot of Metrics
# -----------------------------
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

accuracy = report["accuracy"]
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1 = report["weighted avg"]["f1-score"]

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(6,4))
sns.barplot(x=metrics, y=values, palette="viridis")
plt.ylim(0, 1)
plt.title("Model Evaluation Metrics")
plt.tight_layout()
plt.savefig("results/metrics_barplot.png")
plt.close()

print("\nSaved:")
print("- results/confusion_matrix.png")
print("- results/metrics_barplot.png")
