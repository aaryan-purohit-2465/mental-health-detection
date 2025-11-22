import json
import torch
import yaml
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from .model import MultiTaskClassifier
import torch.nn.functional as F

class SimpleDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.samples = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        text = s.get("text", "")
        mh = int(s.get("label", 0))
        emo = int(s.get("emo_label", 0))
        enc = self.tok(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(mh), torch.tensor(emo)


def train(cfg_path="configs/train_config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg.get("model_name", "distilbert-base-uncased")
    train_path = cfg["data"]["train_path"]
    epochs = int(cfg["training"].get("epochs", 2))
    bs = int(cfg["training"].get("batch_size", 8))
    lr = float(cfg["training"].get("learning_rate", 2e-5))
    max_len = int(cfg["data"].get("max_length", 128))
    outdir = cfg["save"].get("output_dir", "checkpoints/demo_model")
    multitask = bool(cfg.get("multitask", False))
    emo_labels = int(cfg.get("emo_labels", 6))
    mh_labels = int(cfg.get("mh_labels", 3))
    emo_weight = float(cfg.get("emo_weight", 0.5))  # weight for emotion loss

    tok = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskClassifier(model_name=model_name, mh_labels=mh_labels, emo_labels=emo_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = SimpleDataset(train_path, tok, max_len)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    opt = AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for input_ids, attn, mh_label, emo_label in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            mh_label = mh_label.to(device)
            emo_label = emo_label.to(device)

            opt.zero_grad()
            mh_logits, emo_logits = model(input_ids=input_ids, attention_mask=attn)
            mh_loss = F.cross_entropy(mh_logits, mh_label)
            if multitask:
                emo_loss = F.cross_entropy(emo_logits, emo_label)
                loss = mh_loss * (1.0 - emo_weight) + emo_loss * emo_weight
            else:
                loss = mh_loss

            loss.backward()
            opt.step()
            total += loss.item()

        print(f"ep {ep+1}/{epochs} loss {total:.4f}")

    os.makedirs(outdir, exist_ok=True)
    # save only model weights and tokenizer (small enough for demo)
    model_to_save = model.state_dict()
    torch.save(model_to_save, os.path.join(outdir, "pytorch_model.pt"))
    tok.save_pretrained(outdir)
    print("saved", outdir)


if __name__ == "__main__":
    train()
