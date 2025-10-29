import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model
@st.cache_resource
def load_model():
    model_name = "Pulk17/Fake-News-Detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app
st.title("Fake News Detector")
st.write("Paste a news headline or article below to check if it's REAL or FAKE:")

text_input = st.text_area("Enter news text here", height=150)

if st.button("Check Authenticity"):
    if text_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, label].item()
            result = "REAL" if label == 1 else "FAKE"
            st.subheader(f"Prediction: {result}")
            st.write(f"Confidence: {confidence*100:.2f}%")

st.caption("Model: Pulk17/Fake-News-Detection [BERT-base-uncased]")
