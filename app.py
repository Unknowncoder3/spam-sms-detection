#app.py




import pandas as pd
import string
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# Prediction function
def predict_message(message):
    processed = preprocess_text(message)
    vectorized = tfidf.transform([processed]).toarray()
    prediction = model.predict(vectorized)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", page_icon="", layout="centered")

st.title("SMS Spam Classifier")
st.write("Enter an SMS message below to check if it's **Spam** or **Not Spam**.")

user_input = st.text_area("Enter your SMS message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to predict.")
    else:
        result = predict_message(user_input)
        if result == "Spam":
            st.error("This message is classified as **SPAM**!")
        else:
            st.success("This message is classified as **NOT SPAM**!")

st.markdown("---")
st.caption("Built with using Streamlit")
