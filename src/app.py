import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import re
import os
import pickle
from youtube_api import YouTubeAPI
from dotenv import load_dotenv
import time
import pandas as pd
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
stop_words = set(nltk.corpus.stopwords.words('english')) - {'no', 'not', 'hate', 'against', 'racist', 'abuse', 'toxic'}

@st.cache_resource
def load_model():
    """Cargar el modelo desde el archivo h5"""
    model_path = 'final_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encuentra en la ruta: {model_path}")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_tokenizer():
    """Cargar el tokenizer desde el archivo pickle"""
    tokenizer_path = 'tokenizer.pickle'
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"El tokenizer no se encuentra en la ruta: {tokenizer_path}")
    with open(tokenizer_path, 'rb') as handle:
        return pickle.load(handle)

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

def analyze_text(model, tokenizer, text):
    """Analizar un texto y retornar la probabilidad de que sea ofensivo"""
    processed_text = preprocess_text(text)
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded_sequences, verbose=0)
    return float(prediction[0][0])

# Cargar el modelo y el tokenizer
try:
    model = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Error al cargar el modelo o tokenizer: {str(e)}")
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
                prob_ofensivo = analyze_text(model, tokenizer, user_input)
                
                # Crear un contenedor para los resultados
                results_container = st.container()
                
                with results_container:
                    # Mostrar la barra de probabilidad
                    st.progress(prob_ofensivo)
                    
                    # Mostrar el porcentaje y la clasificación
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Probabilidad de mensaje no ofensivo",
                            f"{(1 - prob_ofensivo):.1%}"
                        )
                    with col2:
                        st.metric(
                            "Probabilidad de mensaje ofensivo",
                            f"{prob_ofensivo:.1%}"
                        )
                    
                    # Mostrar la clasificación final
                    if prob_ofensivo > 0.5:
                        st.error('⚠️ Este mensaje ha sido clasificado como potencialmente ofensivo')
                    else:
                        st.success('✅ Este mensaje ha sido clasificado como no ofensivo')
                    
            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
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
                        prob_ofensivo = analyze_text(model, tokenizer, comment)
                        results.append({
                            'texto': comment,
                            'prob_ofensivo': prob_ofensivo
                        })
                        
                        # Actualizar barra de progreso
                        progress_bar.progress((i + 1) / len(comments))
                    
                    # Mostrar resultados
                    st.subheader('Resultados del análisis')
                    
                    # Calcular estadísticas
                    total_offensive = sum(1 for r in results if r['prob_ofensivo'] > 0.5)
                    avg_prob = np.mean([r['prob_ofensivo'] for r in results])
                    
                    # Mostrar resumen
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total de comentarios", len(comments))
                    with col2:
                        st.metric("Comentarios ofensivos", total_offensive)
                    with col3:
                        st.metric("Probabilidad media", f"{avg_prob:.1%}")
                    
                    # Mostrar gráfico de distribución si hay suficientes comentarios
                    if len(results) > 5:
                        probs = [r['prob_ofensivo'] for r in results]
                        st.subheader("Distribución de probabilidades")
                        hist_data = np.histogram(probs, bins=10, range=(0,1))
                        st.bar_chart(pd.DataFrame({
                            'Probabilidad': hist_data[0]
                        }))
                    
                    # Mostrar comentarios individuales
                    st.subheader("Análisis detallado de comentarios")
                    sort_option = st.selectbox(
                        'Ordenar comentarios por:',
                        ['Probabilidad (mayor a menor)', 'Probabilidad (menor a mayor)', 'Orden original']
                    )
                    
                    if sort_option == 'Probabilidad (mayor a menor)':
                        results.sort(key=lambda x: x['prob_ofensivo'], reverse=True)
                    elif sort_option == 'Probabilidad (menor a mayor)':
                        results.sort(key=lambda x: x['prob_ofensivo'])
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"Comentario {i} - Probabilidad de ser ofensivo: {result['prob_ofensivo']:.1%}"
                        ):
                            st.text(result['texto'])
                            st.progress(result['prob_ofensivo'])
                            
            except Exception as e:
                st.error(f"Error inesperado: {str(e)}")
        else:
            st.warning('Por favor, introduce la URL de un video de YouTube.')

# Añadir información sobre el modelo en el sidebar
with st.sidebar:
    st.subheader("Acerca del modelo")
    st.write("""
    Este sistema utiliza un modelo de Deep Learning que combina:
    - Embeddings preentrenados de GloVe
    - Capas convolucionales (CNN)
    - Redes LSTM bidireccionales
    
    El modelo ha sido entrenado para detectar contenido ofensivo 
    en comentarios, incluyendo mensajes de odio, lenguaje abusivo,
    y contenido tóxico.
    
    ⚠️ Nota: Este es un modelo experimental y sus predicciones deben 
    ser consideradas como orientativas.
    """)

