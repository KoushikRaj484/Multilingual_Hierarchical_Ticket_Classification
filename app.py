import re
import os
import nltk
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Note: tokenizer from Keras is not strictly needed for loading, 
# but included for completeness if needed for re-training later.

# --- IMPORTANT: TensorFlow Legacy Loader (Ensures compatibility) ---
# Use TensorFlow's legacy loader for models
load_model = tf.keras.models.load_model

# --- NLTK Configuration for Hugging Face Spaces ---
# HF Spaces use persistent storage, but downloading NLTK data on
# startup is safer for fresh environment builds.
@st.cache_resource
def setup_nltk():
    """Sets up NLTK data and returns English stopwords."""
    # Define a temporary directory for NLTK if needed, 
    # but in HF spaces, it usually works by default or needs a specific path.
    # We will let nltk handle the path for simplicity.
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    return set(stopwords.words("english"))

stop_english = setup_nltk()

# --- File Paths and Loading (CRITICAL for HF Spaces) ---
# Ensure these files are uploaded to your Hugging Face repository 
# alongside this 'app.py' file.
MODEL_PATH = "model_ticket1.h5"
LE_TYPE_PATH = "le_type_ticket.pkl"
LE_QUEUE_PATH = "le_queue_ticket.pkl"
MLB_PATH = "mlb_ticket.pkl"
TOKENIZER_PATH = "tokenizer_ticket.pkl"
MAX_SEQ_LEN = 200  # MUST match training

@st.cache_resource
def load_resources():
    """Loads all model artifacts, including the model and preprocessors."""
    try:
        # Load Model
        # compile=False is necessary if custom objects were not compiled in
        model = load_model(MODEL_PATH, compile=False)
        
        # Load Pickles
        with open(LE_TYPE_PATH, "rb") as f:
            le_type = pickle.load(f)
        with open(LE_QUEUE_PATH, "rb") as f:
            le_queue = pickle.load(f)
        with open(MLB_PATH, "rb") as f:
            mlb = pickle.load(f)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
            
        return model, le_type, le_queue, mlb, tokenizer
        
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure all artifacts (model.h5, *.pkl) are uploaded.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

model, le_type, le_queue, mlb, tokenizer = load_resources()

# --- Text Preprocessing Functions ---

def clean_text(t):
    """Performs text cleaning for a given string."""
    if pd.isna(t) or t is None:
        return ""
    
    t = t.lower()
    # Tokenize and remove stopwords/short words
    tokens = word_tokenize(t)
    tokens = [w for w in tokens if w not in stop_english and len(w) > 2 and w.isalnum()]
    t = " ".join(tokens)
    
    # Regex cleaning (simplified and adjusted)
    # Removing common non-alphanumeric noise, URLs, and emails.
    t = re.sub(r"http\S+|www\.\S+|@\S+|\\n", " ", t)  # URLs, emails, newlines
    # Removing most punctuation but keeping spaces
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t) 
    t = re.sub(r"\s+", " ", t).strip() # Consolidate spaces
    
    return t

def convert_to_sequence(txt):
    """Converts cleaned text to a padded sequence."""
    seq = tokenizer.texts_to_sequences([txt])  # Input must be a list
    padded = pad_sequences(
        seq, maxlen=MAX_SEQ_LEN, padding="pre", truncating="pre"
    )
    return padded

# --- Streamlit UI ---

st.set_page_config(page_title="Ticket Classification")
st.title("🎫 Ticket Classification App")

# Example Text Display
st.header("Example Input")
st.markdown("**Subject:** Account Disruption")
st.code("""Dear Customer Support Team,
I am writing to report a significant problem with the centralized account management portal...""")
st.write("---")

# Input Fields
body = st.text_area("Enter your **Subject** and **Body**:", key="subject_body_input", height=200)
subject = " "
#col1, col2 = st.columns(2)
#with col1:
#    subject = st.text_input("Enter your **Subject**:", key="subject_input")
#with col2:
#    body = st.text_area("Enter your **Body**:", key="body_input", height=100)

# --- Prediction Logic ---

if st.button("Submit"):
    if not subject and not body:
        st.warning("Please enter a subject or body text to classify.")
    else:
        # Combine and Clean
        raw_text =  body + " " + subject 
        cleaned = clean_text(raw_text)
        
        st.subheader("Preprocessing Results")
        st.info(f"**Cleaned Text:** {cleaned}")
        
        # Convert and Predict
        seq = convert_to_sequence(cleaned)
        
        with st.spinner("Classifying ticket..."):
            preds = model.predict(seq, verbose=0)
            
        pred_type_probs, pred_queue_probs, pred_tags_probs = preds
        
        # 1. Decode single-label outputs
        pred_type = le_type.inverse_transform([np.argmax(pred_type_probs)])[0]
        pred_queue = le_queue.inverse_transform([np.argmax(pred_queue_probs)])[0]
        
        # 2. Decode multi-label outputs (Tags)
        pred_tags_binary = (pred_tags_probs >= 0.5).astype(int)
        # mlb.inverse_transform returns a list of tuples, so we take the first element (index 0)
        pred_tags = mlb.inverse_transform(pred_tags_binary)[0]
        
        st.success("✅ Classification Complete!")
        
        #st.subheader("Prediction Results")
        st.metric("Predicted Type", pred_type)
        st.metric("Predicted Queue", pred_queue)
        
        if pred_tags:
            st.markdown(f"**Predicted Tags:** {', '.join(pred_tags)}")
        else:
            st.markdown("**Predicted Tags:** No significant tags found.")