def list_datasets():
    return [
        "CLPsych (request required)",
        "Dreaddit (public dataset)",
        "Synthetic samples (included)"
    ]

def load_synthetic_samples():
    try:
        with open("data/samples/train_sample.jsonl", "r") as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    print("Datasets:")
    for d in list_datasets():
        print(" -", d)
