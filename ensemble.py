import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)

# Public models (no token needed)
bert_name = "bert-base-uncased"
roberta_name = "roberta-base"
flan_t5_name = "google/flan-t5-base"

# Load BERT
bert_tok = AutoTokenizer.from_pretrained(bert_name)
bert = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2)

# Load RoBERTa
roberta_tok = AutoTokenizer.from_pretrained(roberta_name)
roberta = AutoModelForSequenceClassification.from_pretrained(roberta_name, num_labels=2)

# Load FLAN-T5
flan_t5_tok = T5Tokenizer.from_pretrained(flan_t5_name)
flan_t5 = T5ForConditionalGeneration.from_pretrained(flan_t5_name)

def predict_with_bert(text):
    inputs = bert_tok(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()

def predict_with_roberta(text):
    inputs = roberta_tok(text, return_tensors="pt", truncation=True, padding=True)
    outputs = roberta(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()

def predict_with_flan_t5(text):
    prompt = f"Classify as real or fake news: {text}"
    inputs = flan_t5_tok(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = flan_t5.generate(**inputs, max_new_tokens=10)
    result = flan_t5_tok.decode(outputs[0], skip_special_tokens=True).lower()
    return 1.0 if "fake" in result else 0.0

def ensemble_predict(text):
    p1 = predict_with_bert(text)
    p2 = predict_with_roberta(text)
    p3 = predict_with_flan_t5(text)
    avg = (p1 + p2 + p3) / 3
    label = "Fake News" if avg >= 0.5 else "Real News"
    return label, round(avg * 100, 2)
