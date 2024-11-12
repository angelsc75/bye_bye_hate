import streamlit as st
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
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

# Inicializar lematizador y palabras vacías
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesar el texto eliminando caracteres especiales, palabras vacías y lematizando."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def create_additional_features(text):
    """Crear características adicionales para mantener consistencia con el entrenamiento."""
    features = pd.DataFrame()
    features['comment_length'] = [len(text.split())]
    features['hate_frequency'] = [text.lower().split().count('hate')]
    return features

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

# Definir las rutas del modelo, vectorizador y PCA
model_path = './models/LogisticRegression.pkl'
vectorizer_path = './models/LogisticRegression_vectorizer.pkl'
pca_path = './models/pca.pkl'

# Verificar si existen los archivos requeridos
required_files = {
    'Model': model_path,
    'Vectorizer': vectorizer_path,
    'PCA': pca_path
}

for name, path in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {name} no se encuentra en la ruta: {path}")

# Cargar el modelo, vectorizador y PCA
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
pca = joblib.load(pca_path)

# Obtener la clave de la API de YouTube desde las variables de entorno
api_key = os.getenv('YOUTUBE_API_KEY')

st.title('Detección de Mensajes de Odio en Comentarios de YouTube')
user_input = st.text_area('Introduce un comentario', key='user_input')
video_url = st.text_input('Introduce la URL de un video de YouTube', key='video_url')

# Inicializar el estado de la sesión para almacenar resultados
if 'comment_result' not in st.session_state:
    st.session_state.comment_result = None
if 'video_results' not in st.session_state:
    st.session_state.video_results = []

if st.button('Analizar Comentario'):
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
            probabilities = model.predict_proba(user_input_pca)[0]
            
            # Guardar el resultado en el estado de la sesión
            st.session_state.comment_result = {
                'texto': user_input,
                'resultado': resultado,
                'prob_no_ofensivo': probabilities[0],
                'prob_odio': probabilities[1]
            }
            
        except Exception as e:
            st.error(f"Error en el procesamiento: {str(e)}")
    else:
        st.warning('Por favor, introduce un comentario para analizar.')

if st.session_state.comment_result:
    st.write(f"**Texto:** {st.session_state.comment_result['texto']}")
    st.write(f"**Clasificación:** {st.session_state.comment_result['resultado']}")
    st.write(f"**Prob. no ofensivo:** {st.session_state.comment_result['prob_no_ofensivo']:.2%}")
    st.write(f"**Prob. de odio:** {st.session_state.comment_result['prob_odio']:.2%}")

if st.button('Analizar Video'):
    if video_url:
        try:
            # Extraer el ID del video de la URL
            if 'v=' not in video_url:
                st.error('URL de video inválida. Asegúrate de que sea una URL válida de YouTube')
                st.stop()
                
            video_id = video_url.split('v=')[1]
            ampersand_position = video_id.find('&')
            if ampersand_position != -1:
                video_id = video_id[:ampersand_position]
            
            # Verificar la clave de API
            api_key = os.getenv('YOUTUBE_API_KEY')
            if not api_key:
                st.error("Clave de API no encontrada. Asegúrate de que YOUTUBE_API_KEY esté en tu archivo .env")
                st.stop()
            
            # Inicializar el cliente de YouTube
            youtube_api = YouTubeAPI(api_key)
            
            # Mostrar mensaje de carga
            with st.spinner('Obteniendo comentarios del video...'):
                try:
                    comments = youtube_api.get_comments(video_id)
                except Exception as e:
                    st.warning(f"Error al obtener comentarios con la API de YouTube: {str(e)}")
                    st.warning("Intentando obtener comentarios con Selenium...")
                    comments = get_comments_selenium(video_url)
                
                if comments:
                    st.success(f'Se encontraron {len(comments)} comentarios.')
                    
                    # Guardar los resultados en el estado de la sesión
                    st.session_state.video_results = []
                    
                    for i, comment in enumerate(comments, 1):
                        # Preprocesar el comentario
                        comment_cleaned = preprocess_text(comment)
                        comment_vec = vectorizer.transform([comment_cleaned])
                        comment_pca = pca.transform(comment_vec.toarray())
                        
                        # Hacer la predicción
                        prediction = model.predict(comment_pca)
                        probabilities = model.predict_proba(comment_pca)[0]
                        
                        # Guardar el resultado
                        st.session_state.video_results.append({
                            'texto': comment,
                            'resultado': 'Mensaje de odio 😠' if prediction[0] == 1 else 'Mensaje no ofensivo 😊',
                            'prob_no_ofensivo': probabilities[0],
                            'prob_odio': probabilities[1]
                        })
        
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error inesperado: {str(e)}")
            st.error("Por favor, verifica la URL del video y tu conexión a internet")
    else:
        st.warning('Por favor, introduce la URL de un video de YouTube.')

# Mostrar los resultados de los comentarios del video
if st.session_state.video_results:
    for i, result in enumerate(st.session_state.video_results, 1):
        with st.expander(f"Comentario {i}"):
            st.write(f"**Texto:** {result['texto']}")
            st.write(f"**Clasificación:** {result['resultado']}")
            st.write(f"**Prob. no ofensivo:** {result['prob_no_ofensivo']:.2%}")
            st.write(f"**Prob. de odio:** {result['prob_odio']:.2%}")


