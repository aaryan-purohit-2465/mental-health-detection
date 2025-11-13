import streamlit as st

st.title("Mental Health Detection Demo (Prototype)")

text = st.text_area("Enter text to analyze:")

if text:
    st.write("Prediction will appear here (model not added yet).")
    st.write("Anonymized text:")
    st.write(text.lower())
