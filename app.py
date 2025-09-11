import streamlit as st
from ensemble import ensemble_predict

st.set_page_config(page_title="Fake News Ensemble Detector", layout="centered")
st.title("📰 Fake News Ensemble Detector")

st.write("Enter a news headline or paragraph to check if it's **real** or **fake**.")

text = st.text_area("News Text", height=150)

if st.button("Analyze"):
    if text.strip():
        with st.spinner("Analyzing with multiple AI models..."):
            label, confidence = ensemble_predict(text)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence}%")
    else:
        st.warning("Please enter some text to analyze.")
