import streamlit as st
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# Define paths for the model and vectorizer
model_path = './models/LogisticRegression.pkl'
vectorizer_path = './models/LogisticRegression_vectorizer.pkl'

# Verify if the model and vectorizer files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {model_path}")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"El archivo del vectorizador no se encuentra en la ruta: {vectorizer_path}")

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.title('Detección de Mensajes de Odio en Comentarios de YouTube')
user_input = st.text_area('Introduce un comentario')

if st.button('Analizar'):
    user_input_cleaned = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input_cleaned])
    if user_input_vec.shape[1] != model.n_features_in_:
        st.write("Error: El número de características en los datos de entrada no coincide con el número de características con las que se entrenó el modelo.")
    else:
        prediction = model.predict(user_input_vec)
        st.write('Mensaje de odio' if prediction[0] == 1 else 'Mensaje no ofensivo')