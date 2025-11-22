import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", mh_labels=3, emo_labels=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hid = self.bert.config.hidden_size

        # heads
        self.mh_head = nn.Linear(hid, mh_labels)
        self.emo_head = nn.Linear(hid, emo_labels)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        rep = out.last_hidden_state[:, 0]
        mh_logits = self.mh_head(rep)
        emo_logits = self.emo_head(rep)
        return mh_logits, emo_logits


if __name__ == "__main__":
    print("multi task model ready")
