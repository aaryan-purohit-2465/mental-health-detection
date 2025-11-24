import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model & tokenizer (demo_model)
MODEL_PATH = "checkpoints/demo_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Mental Health Detection Demo")

user_input = st.text_area("Enter text:", "")

if st.button("Analyze"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        st.write("### Results")
        st.json({
            "label_0": float(probs[0]),
            "label_1": float(probs[1])
        })
    else:
        st.warning("Please enter some text first.")
