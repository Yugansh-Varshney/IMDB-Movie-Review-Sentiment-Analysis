import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load your pre-trained model
model = load_model("simpleRNN_IMDB.keras")

# Utility: decode integer-encoded review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess raw text into model input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as **Negative** or **Positive**.')

# Text area for user input
user_input = st.text_area('✍️ Movie Review')

if st.button('Classify'):
    if user_input.strip():  # avoid empty input
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]

        sentiment = '😊 Positive' if score > 0.5 else '😡 Negative'
        confidence = score if score > 0.5 else 1 - score

        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Confidence:** {confidence * 100:.2f}%')
    else:
        st.warning('⚠️ Please enter a movie review before classifying.')
