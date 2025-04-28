# main.py

import pandas as pd
import string
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# Download NLTK stopwords (only once)
nltk.download('stopwords')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load datasets
data1 = pd.read_csv('spam.csv', encoding='latin-1')
data1 = data1[['v1', 'v2']]
data1.columns = ['label', 'message']

data2 = pd.read_csv('sample_sms_messages.csv')

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1}).fillna(data['label']).astype(int)

# Preprocess messages
data['processed_message'] = data['message'].apply(preprocess_text)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=7000)
X = tfidf.fit_transform(data['processed_message']).toarray()
y = data['label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if model exists to avoid retraining every time
if not os.path.exists('spam_classifier_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Streamlit App
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message below to check if it's **Spam** or **Not Spam**.")

def predict_message(message):
    processed = preprocess_text(message)
    vectorized = tfidf.transform([processed]).toarray()
    prediction = model.predict(vectorized)
    probability = model.predict_proba(vectorized).max()
    return ('Spam' if prediction[0] == 1 else 'Not Spam'), probability

user_input = st.text_area("Enter your SMS message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to predict.")
    else:
        result, prob = predict_message(user_input)
        if result == "Spam":
            st.error(f"🚫 This message is classified as **SPAM**! (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ This message is classified as **NOT SPAM**! (Confidence: {prob:.2f})")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
