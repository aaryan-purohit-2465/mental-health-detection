import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "checkpoints/demo_model"

LABELS = {
    0: "Neutral",
    1: "Possible Mental Distress"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="Mental Health Text Classifier", layout="centered")
st.title("Mental Health Text Classifier")
st.write("Enter a short text and the model will try to guess if it shows signs of mental distress.")

tokenizer, model = load_model()

text = st.text_area("Write something here:", height=140)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=1)[0]

        pred_index = int(torch.argmax(probs))
        pred_label = LABELS.get(pred_index, "Unknown")

        st.subheader("Prediction")
        st.markdown(f"**{pred_label}**")

        st.subheader("Confidence")
        st.progress(float(probs[pred_index]))

        # show numeric scores without using a dict or dataframe
        st.write("Scores (probabilities):")
        st.write(f"- Neutral: {probs[0].item():.3f}")
        st.write(f"- Possible Mental Distress: {probs[1].item():.3f}")
