import streamlit as st
from ensemble import ensemble_predict

st.set_page_config(page_title="Fake News Ensemble Detector", layout="centered")
st.title("📰 Fake News Ensemble Detector")
st.write("Ultra-accurate fake news detection using 5 powerful transformer models.")

text = st.text_area("Enter news text:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter news text.")
    else:
        label, prob = ensemble_predict(text)
        icon = "🟩" if label=="Real" else "🟥"
        st.subheader(f"{icon} {label} News")
        st.write(f"Confidence: **{(prob if prob>0.5 else 1-prob)*100:.2f}%**")
