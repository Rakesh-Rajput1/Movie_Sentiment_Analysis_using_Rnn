import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load word index
word_index = imdb.get_word_index()

# Create reverse word index
reverse_word_index = {value: key for (key, value) in word_index.items()}

## Load model
model = load_model('./models/simple_rnn_imdb.h5')
model.summary()

# To get weights
weights = model.get_weights()


def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encode_review])


def preprocess_text(text):
    words=text.lower().split()
    encode_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encode_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment, prediction[0][0]


## streamlit app 

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')


## user input
user_input=st.text_area('Movie_Review')


if st.button('classify'):
    preprocess_input=preprocess_text(user_input)


## make prediction 
    prediction=model.predict(preprocess_input)
    sentiment=model.predict(preprocess_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'Negative'


    # Display the result
    st.write(f'sentiment: {sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please Enter a movie review')