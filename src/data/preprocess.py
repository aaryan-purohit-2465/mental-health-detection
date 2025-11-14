import re

def anonymize_text(text: str) -> str:
    """
    Removes usernames, URLs, emails, and numbers.
    """
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    return text.lower()

if __name__ == "__main__":
    sample = "Feeling sad today... contact me @someone or visit http://example.com"
    print(anonymize_text(sample))
