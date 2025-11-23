import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "checkpoints/demo_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

LABEL_MAP = {
    0: "neutral",
    1: "depression",
    2: "anxiety"
}

def predict(text):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**tokens)
        probs = torch.softmax(out.logits, dim=-1).cpu().tolist()

    results = []
    for p in probs:
        pred_idx = int(torch.tensor(p).argmax().item())
        results.append({
            "probs": {f"label_{i}": float(p[i]) for i in range(len(p))},
            "pred_label": pred_idx,
            "pred_name": LABEL_MAP.get(pred_idx, str(pred_idx))
        })
    return results[0] if len(results) == 1 else results

if __name__ == "__main__":
    s = "I feel exhausted and empty these days."
    print(predict(s))
