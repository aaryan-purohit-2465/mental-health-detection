from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """
    Returns a DistilBERT tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    print("Tokenizer loaded. Example:", tokenizer("Hello World"))
