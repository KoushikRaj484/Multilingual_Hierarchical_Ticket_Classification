import re
import nltk
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# -----------------------------
# NLTK Requirements
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

stop_english = set(stopwords.words("english"))

# -----------------------------
# Example text for display
# -----------------------------
st.write("Account Disruption")
st.write("""Dear Customer Support Team,

I am writing to report a significant problem with the centralized account management portal, 
which currently appears to be offline. 
This outage is blocking access to account settings...

""")

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Ticket Classification App")

col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Enter your subject:")
with col2:
    body = st.text_input("Enter your body:")

# -----------------------------
# Load Model
# -----------------------------
model_path = "model.h5"
model = load_model(model_path)

# -----------------------------
# Load Tokenizer (IMPORTANT)
# -----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_SEQ_LEN = 107   # ← Must match training


# -----------------------------
# Clean Text
# -----------------------------
def clean_text(t):
    if pd.isna(t):
        return ""

    t = t.lower()
    tokens = word_tokenize(t)
    tokens = [w for w in tokens if w not in stop_english and len(w) > 2]
    t = " ".join(tokens)

    # regex cleaning
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"\\n", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"[%\[\]_\\<\(\]#\?\'\":\)\-\;\+\!\/,>\.\n\r]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t


# -----------------------------
# Convert Text → Sequence
# -----------------------------
def convert_to_sequence(txt):
    seq = tokenizer.texts_to_sequences([txt])  # must be list
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="pre", truncating="pre")
    return padded



# -----------------------------
# Prediction
# -----------------------------
if st.button("Submit"):
    # combine subject & body
    raw_text = subject + " " + body

    cleaned = clean_text(raw_text)
    st.write("Cleaned Text:", cleaned)

    seq = convert_to_sequence(cleaned)

    preds = model.predict(seq)
    st.write("Model Output:", preds)
