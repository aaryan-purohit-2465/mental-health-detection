"""
Placeholder functions for dataset loading.

This file does NOT download or store any raw social media data.
It only provides a structure for safe dataset handling.
"""

def list_datasets():
    """
    Returns a list of datasets this project can work with.
    (These datasets are NOT included in this repo.)
    """
    return [
        "CLPsych (Reddit) — requires request",
        "Dreaddit — public dataset",
        "Synthetic samples (included in data/samples)"
    ]


def load_synthetic_samples():
    """
    Loads the safe synthetic samples from data/samples/.
    """
    import json

    try:
        with open("data/samples/train_sample.jsonl", "r") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    print("Available datasets:")
    for d in list_datasets():
        print(" -", d)

    print("\nSynthetic sample count:", len(load_synthetic_samples()))
