import json
from collections import Counter
import re
from pathlib import Path

DATA_PATH = Path("data/samples/train_sample.jsonl")

def simple_tokenize(text):
    # lowercase, remove punctuation, split on spaces
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t.strip()]

def load_samples(path=DATA_PATH):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

def label_counts(samples):
    c = Counter()
    for s in samples:
        c[s.get("label")] += 1
    return c

def top_tokens(samples, top_k=20):
    c = Counter()
    for s in samples:
        tokens = simple_tokenize(s.get("text", ""))
        c.update(tokens)
    return c.most_common(top_k)

if __name__ == "__main__":
    samples = load_samples()
    print(f"Loaded {len(samples)} samples\n")
    counts = label_counts(samples)
    print("Label counts:")
    for label, cnt in counts.items():
        print(f"  {label}: {cnt}")
    print("\nTop tokens:")
    for tok, freq in top_tokens(samples, top_k=30):
        print(f"  {tok}: {freq}")
