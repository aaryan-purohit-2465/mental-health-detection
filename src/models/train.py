import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import yaml
import os

class MentalHealthDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.samples = [json.loads(line) for line in open(jsonl_path, "r")]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        label = item["label"]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


def train(config_path="configs/train_config.yaml"):
    # Load config
    cfg = yaml.safe_load(open(config_path))

    model_name = cfg["model_name"]
    train_path = cfg["data"]["train_path"]

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    max_length = cfg["data"]["max_length"]
    output_dir = cfg["save"]["output_dir"]

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )

    dataset = MentalHealthDataset(train_path, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Training on {device} for {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train()
