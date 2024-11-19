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
import pymongo
from datetime import datetime
import threading
import csv
import schedule


# Cargar variables de entorno
load_dotenv()

# Ahora puedes acceder a las variables de entorno
tf_enable_onednn_opts = os.getenv('TF_ENABLE_ONEDNN_OPTS')
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
    model_path = 'models/final_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encuentra en la ruta: {model_path}")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_tokenizer():
    """Cargar el tokenizer desde el archivo pickle"""
    tokenizer_path = 'models/tokenizer.pickle'
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

# MongoDB configuration
def init_mongodb():
    """Initialize MongoDB connection"""
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = pymongo.MongoClient(mongo_uri)
    db = client[os.getenv('DATABASE_NAME', 'NLP')]
    return db[os.getenv('COLLECTION_NAME', 'lebels')]

def save_to_mongodb(comments_data, video_id):
    """Save only toxic comments (>50% probability) to MongoDB"""
    try:
        collection = init_mongodb()
        toxic_comments = [
            comment for comment in comments_data
            if comment['prob_ofensivo'] > 0.5
        ]

        for comment in toxic_comments:
            comment['video_id'] = video_id
            comment['timestamp'] = datetime.now()
            collection.update_one(
                {'texto': comment['texto'], 'video_id': video_id},
                {'$set': comment},
                upsert=True
            )

        return len(toxic_comments)
    except Exception as e:
        st.error(f"Error saving to MongoDB: {str(e)}")
        return 0

def export_to_csv(comments_data, video_id):
    """Export comments to CSV file"""
    try:
        filename = f'comments_{video_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df = pd.DataFrame(comments_data)
        df['timestamp'] = datetime.now()
        df['video_id'] = video_id
        df.to_csv(filename, index=False)
        return filename
    except Exception as e:
        st.error(f"Error exporting to CSV: {str(e)}")
        return None

def schedule_updates(video_id):
    """Schedule periodic updates of comments"""
    def update_comments():
        try:
            youtube_api = YouTubeAPI(os.getenv('YOUTUBE_API_KEY'))
            comments = youtube_api.get_comments(video_id)
            results = []

            for comment in comments:
                prob_ofensivo = analyze_text(model, tokenizer, comment)
                results.append({
                    'texto': comment,
                    'prob_ofensivo': prob_ofensivo
                })

            save_to_mongodb(results, video_id)
            print(f"Updated comments for video {video_id} at {datetime.now()}")
        except Exception as e:
            print(f"Error updating comments: {str(e)}")

    schedule.every(30).seconds.do(update_comments)

    while True:
        schedule.run_pending()
        time.sleep(1)

# Cargar el modelo y el tokenizer
try:
    model = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Error al cargar el modelo o tokenizer: {str(e)}")
    st.stop()

def main():
    st.title('Detección de Mensajes de Odio en Comentarios de YouTube')

    # Crear las pestañas
    tab1, tab2, tab3 = st.tabs(["Análisis de Comentario", "Análisis de Video", "Exportar y Monitorear"])

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
                        elif sort_option == 'Probabilidad (menor a menor)':
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

    with tab3:
        st.subheader("Exportación y Monitoreo de Comentarios")

        # MongoDB connection status
        try:
            collection = init_mongodb()
            st.success("✅ Conexión a MongoDB establecida")
        except Exception as e:
            st.error(f"❌ Error de conexión a MongoDB: {str(e)}")

        # Video URL input for monitoring
        monitor_url = st.text_input('URL del video para monitorear', key='monitor_url')

        if monitor_url:
            video_id = monitor_url.split('v=')[1].split('&')[0]

            col1, col2 = st.columns(2)

            with col1:
                if st.button('Iniciar Monitoreo'):
                    try:
                        # Start monitoring in a separate thread
                        monitor_thread = threading.Thread(
                            target=schedule_updates,
                            args=(video_id,),
                            daemon=True
                        )
                        monitor_thread.start()
                        st.success("Monitoreo iniciado - Actualizando cada 0.5 minutos")
                    except Exception as e:
                        st.error(f"Error al iniciar monitoreo: {str(e)}")

            with col2:
                if st.button('Exportar a CSV'):
                    try:
                        collection = init_mongodb()
                        comments = list(collection.find({'video_id': video_id}))
                        if comments:
                            filename = export_to_csv(comments, video_id)
                            if filename:
                                st.success(f"Comentarios exportados a {filename}")

                                # Add download button
                                with open(filename, 'rb') as f:
                                    st.download_button(
                                        label="Descargar CSV",
                                        data=f,
                                        file_name=filename,
                                        mime='text/csv'
                                    )
                        else:
                            st.warning("No hay comentarios guardados para este video")
                    except Exception as e:
                        st.error(f"Error al exportar comentarios: {str(e)}")

        # Display stored comments
        if st.checkbox("Ver comentarios almacenados"):
            try:
                collection = init_mongodb()
                stored_videos = collection.distinct('video_id')

                if stored_videos:
                    selected_video = st.selectbox(
                        "Seleccionar video:",
                        options=stored_videos
                    )

                    comments = list(collection.find({'video_id': selected_video}))
                    if comments:
                        df = pd.DataFrame(comments)
                        # Reordenar y seleccionar columnas específicas
                        columns_to_show = ['texto', 'prob_ofensivo', 'timestamp', 'video_id']
                        df = df[columns_to_show]
                        # Formatear la columna timestamp
                        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        # Mostrar el DataFrame con formato
                        st.dataframe(
                            df.style.format({
                                'prob_ofensivo': '{:.1%}'
                            }),
                            use_container_width=True
                        )

                        # Añadir botón para exportar los comentarios mostrados
                        if st.button('Exportar comentarios mostrados'):
                            filename = export_to_csv(comments, selected_video)
                            if filename:
                                st.success(f"Comentarios exportados a {filename}")
                                with open(filename, 'rb') as f:
                                    st.download_button(
                                        label="Descargar CSV",
                                        data=f,
                                        file_name=filename,
                                        mime='text/csv'
                                    )
                    else:
                        st.info("No hay comentarios para este video")
                else:
                    st.info("No hay videos monitoreados en la base de datos")
            except Exception as e:
                st.error(f"Error al cargar comentarios: {str(e)}")

        # Añadir sección de estadísticas
        if st.checkbox("Ver estadísticas generales"):
            try:
                collection = init_mongodb()
                # Primero verificamos si hay datos
                if collection.count_documents({}) > 0:
                    latest_doc = collection.find_one(sort=[('timestamp', -1)])
                    stats = {
                        'total_videos': len(collection.distinct('video_id')),
                        'total_comments': collection.count_documents({}),
                        'offensive_comments': collection.count_documents({'prob_ofensivo': {'$gt': 0.5}}),
                        'latest_update': latest_doc['timestamp'] if latest_doc else datetime.now()
                    }

                    # Mostrar estadísticas en columnas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Videos", stats['total_videos'])
                    with col2:
                        st.metric("Total Comentarios", stats['total_comments'])
                    with col3:
                        st.metric("Comentarios Ofensivos", stats['offensive_comments'])
                    with col4:
                        st.metric("Última Actualización",
                                 stats['latest_update'].strftime('%Y-%m-%d %H:%M'))

                    # Mostrar gráfico de tendencia temporal si hay suficientes datos
                    pipeline = [
                        {
                            '$group': {
                                '_id': {
                                    '$dateToString': {
                                        'format': '%Y-%m-%d',
                                        'date': '$timestamp'
                                    }
                                },
                                'avg_offensive': {'$avg': '$prob_ofensivo'},
                                'count': {'$sum': 1}
                            }
                        },
                        {'$sort': {'_id': 1}}
                    ]
                    trend_data = list(collection.aggregate(pipeline))

                    if trend_data:
                        trend_df = pd.DataFrame(trend_data)
                        trend_df.columns = ['fecha', 'promedio_ofensivo', 'cantidad']

                        st.subheader("Tendencia temporal de comentarios ofensivos")
                        st.line_chart(trend_df.set_index('fecha')['promedio_ofensivo'])
                else:
                    st.info("No hay datos almacenados en la base de datos todavía. Monitorea algunos videos para ver las estadísticas.")

            except Exception as e:
                st.error(f"Error al cargar estadísticas: {str(e)}")

if __name__ == "__main__":
    main()