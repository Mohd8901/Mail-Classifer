import streamlit as st
import joblib
import re

st.title('Spam Classifier By naive Bayes Model')

st.write('by mohd')

model = joblib.load('Spam_model.h5')
vectorizer = joblib.load('Vectorizer.h5')

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.strip()  # Remove leading/trailing whitespaces
    return text

def predict_email(content):
    cleaned_content = clean_text(content)  # Clean the email content
    vectorized_content = vectorizer.transform([cleaned_content])  # Vectorize the content
    prediction = model.predict(vectorized_content)  # Predict spam or ham
    return 'spam' if prediction[0] == 1 else 'ham'

text = st.text_area('Enter You Mail Content To Classify')

if st.button('Predict'):
    if text:
        result = predict_email(text)
        if result == 1:
            st.error('The Given Mail is Predicted As SPAM')
        else:
            st.success('The Given Mail is Predicted As HAM')
    else:
        st.warning('Enter Some Text To Classify')