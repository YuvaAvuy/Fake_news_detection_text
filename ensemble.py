import torch, numpy as np, tensorflow as tf
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          T5Tokenizer, T5ForConditionalGeneration)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
bert1_name = "jy46604790/Fake-News-Bert-Detect"
bert2_name = "omykhailiv/bert-fake-news-recognition"
roberta_name = "hamzab/roberta-fake-news-classification-dl"

bert1_tok = AutoTokenizer.from_pretrained(bert1_name)
bert1 = AutoModelForSequenceClassification.from_pretrained(bert1_name).to(device)

bert2_tok = AutoTokenizer.from_pretrained(bert2_name)
bert2 = AutoModelForSequenceClassification.from_pretrained(bert2_name).to(device)

roberta_tok = AutoTokenizer.from_pretrained(roberta_name)
roberta = AutoModelForSequenceClassification.from_pretrained(roberta_name).to(device)

flan_t5_tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
flan_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

flan_ul2_tok = T5Tokenizer.from_pretrained("google/flan-ul2-llm")
flan_ul2 = T5ForConditionalGeneration.from_pretrained("google/flan-ul2-llm").to(device)

def get_bert_pred(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    return probs[0][1].item()

def get_flan_pred(text, tokenizer, model):
    prompt = f"Classify this news as Real or Fake:\n{text}\nAnswer only 'Real' or 'Fake'."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return 1.0 if "real" in out else 0.0

def get_all_predictions(text):
    scores = []
    scores.append(get_bert_pred(text, bert1_tok, bert1))
    scores.append(get_bert_pred(text, bert2_tok, bert2))
    scores.append(get_bert_pred(text, roberta_tok, roberta))
    scores.append(get_flan_pred(text, flan_t5_tok, flan_t5))
    scores.append(get_flan_pred(text, flan_ul2_tok, flan_ul2))
    return scores

# Simple meta classifier
meta_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Warmup dummy training (just to initialize weights)
X_dummy = np.random.rand(100,5)
y_dummy = np.random.randint(0,2,100)
meta_model.fit(X_dummy, y_dummy, epochs=3, verbose=0)

def ensemble_predict(text):
    scores = np.array(get_all_predictions(text)).reshape(1, -1)
    prob = float(meta_model.predict(scores, verbose=0)[0][0])
    label = "Real" if prob > 0.5 else "Fake"
    return label, prob
