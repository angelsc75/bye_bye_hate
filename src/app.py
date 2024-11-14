import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import nltk
from youtube_api import YouTubeAPI
from dotenv import load_dotenv
import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Cargar variables de entorno
load_dotenv()

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración del modelo
MAX_WORDS = 15000
MAX_LEN = 128

# Inicializar lematizador y palabras vacías
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'no', 'not', 'hate', 'against', 'racist', 'abuse', 'toxic'}

@st.cache_resource
def load_model():
    """Cargar el modelo desde el archivo h5"""
    model_path = 'models/final_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encuentra en la ruta: {model_path}")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def create_tokenizer():
    """Crear y configurar el tokenizer"""
    return Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')

def preprocess_text(text):
    """Preprocesar el texto con el mismo proceso usado en el entrenamiento"""
    if not isinstance(text, str):
        text = str(text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Preservar ciertas palabras compuestas ofensivas
    text = text.replace('son of a bitch', 'sonofabitch')
    text = text.replace('f u c k', 'fuck')
    text = text.replace('b i t c h', 'bitch')
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Preservar algunos caracteres especiales que pueden indicar toxicidad
    text = re.sub(r'[^a-zA-Z\s!?*#@$]', '', text)
    
    # Eliminar espacios extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lematización y eliminación de stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def get_comments_selenium(video_url):
    """Obtener comentarios de un video de YouTube usando Selenium"""
    data = []
    service = Service(ChromeDriverManager().install())
    with Chrome(service=service) as driver:
        wait = WebDriverWait(driver, 15)
        driver.get(video_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        for _ in range(5):
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(3)
        comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer #content")))
        for comment in comments:
            try:
                comment_text = comment.find_element(By.CSS_SELECTOR, "#content-text").text
                data.append(comment_text)
            except Exception as e:
                print(f"Error al obtener comentario: {e}")
    return data

# Cargar el modelo
try:
    model = load_model()
    tokenizer = create_tokenizer()
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

st.title('Detección de Mensajes de Odio en Comentarios de YouTube')
st.caption('Powered by Deep Learning')

# Crear las pestañas
tab1, tab2 = st.tabs(["Análisis de Comentario", "Análisis de Video"])

with tab1:
    user_input = st.text_area('Introduce un comentario para analizar', key='user_input')
    
    if st.button('Analizar Comentario', key='analyze_comment'):
        if user_input:
            try:
                # Preprocesar el texto
                processed_text = preprocess_text(user_input)
                
                # Tokenizar y padear la secuencia
                sequences = tokenizer.texts_to_sequences([processed_text])
                padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
                
                # Hacer la predicción
                prediction = model.predict(padded_sequences)
                
                # Mostrar resultados con una barra de progreso
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Probabilidad de mensaje no ofensivo",
                        f"{(1 - prediction[0][0]):.2%}"
                    )
                    
                with col2:
                    st.metric(
                        "Probabilidad de mensaje ofensivo",
                        f"{prediction[0][0]:.2%}"
                    )
                
                # Mostrar una barra de progreso
                st.progress(float(prediction[0][0]))
                
                # Mostrar el resultado final
                if prediction[0][0] > 0.5:
                    st.error('⚠️ Este comentario ha sido clasificado como potencialmente ofensivo')
                else:
                    st.success('✅ Este comentario parece no ser ofensivo')
                
            except Exception as e:
                st.error(f"Error en el procesamiento: {str(e)}")
        else:
            st.warning('Por favor, introduce un comentario para analizar.')

with tab2:
    video_url = st.text_input('Introduce la URL de un video de YouTube', key='video_url')
    
    if st.button('Analizar Video', key='analyze_video'):
        if video_url:
            try:
                # Extraer el ID del video de la URL
                video_id = video_url.split('v=')[1].split('&')[0]
                
                # Verificar la clave de API
                api_key = os.getenv('YOUTUBE_API_KEY')
                if not api_key:
                    st.error("Clave de API no encontrada. Asegúrate de que YOUTUBE_API_KEY esté en tu archivo .env")
                    st.stop()
                
                # Inicializar el cliente de YouTube
                youtube_api = YouTubeAPI(api_key)
                
                # Obtener comentarios
                with st.spinner('Obteniendo comentarios del video...'):
                    try:
                        comments = youtube_api.get_comments(video_id)
                    except Exception as e:
                        st.warning(f"Error al obtener comentarios con la API de YouTube: {str(e)}")
                        st.info("Intentando obtener comentarios con Selenium...")
                        comments = get_comments_selenium(video_url)
                
                if comments:
                    st.success(f'Se encontraron {len(comments)} comentarios')
                    
                    # Analizar comentarios
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, comment in enumerate(comments):
                        # Preprocesar y predecir
                        processed_text = preprocess_text(comment)
                        sequences = tokenizer.texts_to_sequences([processed_text])
                        padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
                        prediction = model.predict(padded_sequences, verbose=0)
                        
                        results.append({
                            'texto': comment,
                            'prob_ofensivo': float(prediction[0][0])
                        })
                        
                        # Actualizar barra de progreso
                        progress_bar.progress((i + 1) / len(comments))
                    
                    # Mostrar resultados
                    st.subheader('Resultados del análisis')
                    
                    # Calcular estadísticas
                    total_offensive = sum(1 for r in results if r['prob_ofensivo'] > 0.5)
                    
                    # Mostrar resumen
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total de comentarios", len(comments))
                    with col2:
                        st.metric("Comentarios ofensivos", total_offensive)
                    with col3:
                        st.metric("Porcentaje ofensivo", f"{(total_offensive/len(comments)):.1%}")
                    
                    # Mostrar comentarios individuales
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Comentario {i}"):
                            st.text(result['texto'])
                            st.progress(result['prob_ofensivo'])
                            if result['prob_ofensivo'] > 0.5:
                                st.error(f"Probabilidad de ser ofensivo: {result['prob_ofensivo']:.2%}")
                            else:
                                st.success(f"Probabilidad de ser ofensivo: {result['prob_ofensivo']:.2%}")
                                
            except Exception as e:
                st.error(f"Error inesperado: {str(e)}")
        else:
            st.warning('Por favor, introduce la URL de un video de YouTube.')

# Añadir información sobre el modelo en el sidebar
with st.sidebar:
    st.subheader("Acerca del modelo")
    st.write("""
    Este sistema utiliza un modelo de Deep Learning basado en redes neuronales LSTM 
    bidireccionales con embeddings pre-entrenados para detectar contenido ofensivo 
    en comentarios.
    
    El modelo ha sido entrenado para detectar:
    - Mensajes de odio
    - Contenido abusivo
    - Lenguaje ofensivo
    - Contenido provocativo
    
    ⚠️ Nota: Este es un modelo experimental y sus predicciones deben ser consideradas 
    como orientativas.
    """)


