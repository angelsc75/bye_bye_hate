import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.utils.class_weight import compute_class_weight
import optuna
import logging
import spacy
import gensim.downloader as api
import pickle

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Descargar recursos NLTK
nltk.download('punkt')  # Cambiado de punkt_tab a punkt
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración de parámetros base
MAX_WORDS = 15000
MAX_LEN = 128
EMBEDDING_DIM = 300  # Cambiado a 300 para coincidir con GloVe

def preprocess_text(text_series):
    """
    Preprocesamiento mejorado con manejo especial de palabras ofensivas.
    """
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'hate', 'against', 'racist', 'abuse', 'toxic'}
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
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

    return text_series.apply(clean_text)

def prepare_data(df):
    """
    Prepara los datos para el entrenamiento.
    """
    # Preprocesar texto
    print("Iniciando preprocesamiento de texto...")
    processed_texts = preprocess_text(df['Text'])

    # Tokenización y creación de secuencias
    print("Tokenizando textos...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(processed_texts)
    sequences = tokenizer.texts_to_sequences(processed_texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Preparar etiquetas
    target_columns = ['IsToxic', 'IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']
    df['IsOffensive'] = df[target_columns].any(axis=1)
    y = df['IsOffensive'].astype(int)

    # Cargar los embeddings de GloVe
    print("Cargando embeddings de GloVe...")
    glove_model = api.load("glove-wiki-gigaword-300")

    # Crear matriz de embeddings
    embedding_matrix = np.zeros((MAX_WORDS + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i < MAX_WORDS:  # Solo procesamos hasta MAX_WORDS
            try:
                embedding_matrix[i] = glove_model[word]
            except KeyError:
                continue  # Si la palabra no está en GloVe, dejamos el vector en ceros

    return X, y, tokenizer, embedding_matrix

def create_model_tuned(vocab_size, num_labels, params, embedding_matrix):
    """
    Versión mejorada del modelo con parámetros optimizados y embeddings
    """
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False),
        Conv1D(params.get('conv_filters', 128), 5, activation='relu'),
        Bidirectional(LSTM(params.get('lstm_units_1', 64), return_sequences=True)),
        Bidirectional(LSTM(params.get('lstm_units_2', 32), return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(params.get('dense_units_1', 128), activation='relu'),
        Dropout(params.get('dropout_1', 0.5)),
        Dense(params.get('dense_units_2', 64), activation='relu'),
        Dropout(params.get('dropout_2', 0.3)),
        Dense(num_labels, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

def objective(trial, X, y, tokenizer, embedding_matrix):
    """
    Función objetivo para Optuna
    """
    params = {
        'conv_filters': trial.suggest_int('conv_filters', 64, 256),
        'lstm_units_1': trial.suggest_int('lstm_units_1', 32, 128),
        'lstm_units_2': trial.suggest_int('lstm_units_2', 16, 64),
        'dense_units_1': trial.suggest_int('dense_units_1', 64, 256),
        'dense_units_2': trial.suggest_int('dense_units_2', 32, 128),
        'dropout_1': trial.suggest_float('dropout_1', 0.2, 0.6),
        'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    # Validación cruzada para evaluación más robusta
    kf = KFold(n_splits=3, shuffle=True)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_model_tuned(MAX_WORDS + 1, 1, params, embedding_matrix)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )

        scores.append(history.history['val_accuracy'][-1])

    return np.mean(scores)

def train_with_optimization(df, n_trials=20):
    """
    Pipeline de entrenamiento con optimización de hiperparámetros
    """
    logging.info("Preparando datos...")
    X, y, tokenizer, embedding_matrix = prepare_data(df)

    logging.info("Iniciando optimización de hiperparámetros...")
    study = optuna.create_study(direction='maximize')

    try:
        study.optimize(lambda trial: objective(trial, X, y, tokenizer, embedding_matrix),
                      n_trials=n_trials)

        best_params = study.best_params
        logging.info("Mejores hiperparámetros encontrados:")
        for param, value in best_params.items():
            logging.info(f"{param}: {value}")

        logging.info("Entrenando modelo final...")
        final_model = create_model_tuned(MAX_WORDS + 1, 1, best_params, embedding_matrix)
        history = final_model.fit(
            X, y,
            epochs=15,
            batch_size=best_params.get('batch_size', 32),
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
        )

        return final_model, history

    except Exception as e:
        logging.error(f"Error durante la optimización: {str(e)}")
        default_params = {
            'conv_filters': 128,
            'lstm_units_1': 64,
            'lstm_units_2': 32,
            'dense_units_1': 128,
            'dense_units_2': 64,
            'dropout_1': 0.5,
            'dropout_2': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        logging.info("Usando parámetros por defecto debido al error...")
        final_model = create_model_tuned(MAX_WORDS + 1, 1, default_params, embedding_matrix)
        history = final_model.fit(
            X, y,
            epochs=15,
            batch_size=default_params['batch_size'],
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
        )

        return final_model, history

if __name__ == "__main__":
    import tokenizer
    # Asegúrate de que el archivo CSV existe y tiene las columnas correctas
    df = pd.read_csv('youtoxic_english_1000.csv')

    try:
        logging.info("Iniciando entrenamiento con optimización...")
        final_model, history = train_with_optimization(df, n_trials=20)
        with open('tokenizer.pickle', 'wb') as handle:
          pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Guardar el modelo
        final_model.save('final_model.h5')
        logging.info("Entrenamiento completado y modelo guardado.")

    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")