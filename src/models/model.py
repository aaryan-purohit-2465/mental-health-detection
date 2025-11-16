import torch
import torch.nn as nn
from transformers import AutoModel

class MentalHealthClassifier(nn.Module):
    """
    Basic DistilBERT-based classifier for mental health detection.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Classifier for depression / anxiety / neutral
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]   # CLS token output
        logits = self.classifier(pooled_output)
        return logits


# Quick demo check
if __name__ == "__main__":
    print("Model skeleton created successfully.")
