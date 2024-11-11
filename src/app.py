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
    """
    Preprocesar el texto:
    - Convertir a minúsculas
    - Eliminar URLs
    - Eliminar caracteres especiales y números
    - Eliminar espacios extras
    - Tokenizar, lematizar y eliminar stopwords
    """
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar URLs
    text = re.sub(r'http\\S+|www\\.\\S+', '', text)
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    # Eliminar espacios extra
    text = re.sub(r'\\s+', ' ', text).strip()
    # Tokenizar, lematizar y eliminar stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def create_additional_features(text):
    """Crear características adicionales para mantener consistencia con el entrenamiento"""
    features = pd.DataFrame()
    features['comment_length'] = [len(text.split())]
    features['hate_frequency'] = [text.lower().split().count('hate')]
    return features

# Define paths for the model, vectorizer and PCA
model_path = './models/LogisticRegression.pkl'
vectorizer_path = './models/LogisticRegression_vectorizer.pkl'
pca_path = './models/pca.pkl'

# Verify if all required files exist
required_files = {
    'Model': model_path,
    'Vectorizer': vectorizer_path,
    'PCA': pca_path
}

for name, path in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {name} no se encuentra en la ruta: {path}")

# Load the model, vectorizer and PCA
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
pca = joblib.load(pca_path)

st.title('Detección de Mensajes de Odio en Comentarios de YouTube')
user_input = st.text_area('Introduce un comentario')

if st.button('Analizar'):
    if user_input:
        try:
            # Preprocesar el texto
            user_input_cleaned = preprocess_text(user_input)
            
            # Vectorizar el texto
            user_input_vec = vectorizer.transform([user_input_cleaned])
            
            # Aplicar PCA
            user_input_pca = pca.transform(user_input_vec.toarray())
            
            # Hacer la predicción
            prediction = model.predict(user_input_pca)
            
            # Mostrar resultado
            resultado = 'Mensaje de odio 😠' if prediction[0] == 1 else 'Mensaje no ofensivo 😊'
            st.write(resultado)
            
            # Mostrar probabilidades (opcional)
            probabilities = model.predict_proba(user_input_pca)[0]
            st.write(f'Probabilidad de mensaje no ofensivo: {probabilities[0]:.2%}')
            st.write(f'Probabilidad de mensaje de odio: {probabilities[1]:.2%}')
            
        except Exception as e:
            st.error(f"Error en el procesamiento: {str(e)}")
    else:
        st.warning('Por favor, introduce un comentario para analizar.')