import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración de parámetros
MAX_WORDS = 15000
MAX_LEN = 100
EMBEDDING_DIM = 200

def preprocess_text(text_series):
    """
    Preprocesamiento mejorado con manejo especial de palabras ofensivas
    """
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'hate', 'against'}
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
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
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Eliminar stopwords y lematización, preservando palabras importantes
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    
    return text_series.apply(clean_text)

def prepare_data(df):
    """
    Prepara los datos para el entrenamiento
    """
    # Preprocesar texto
    print("Iniciando preprocesamiento de texto...")
    processed_texts = preprocess_text(df['Text'])
    
    # Tokenización con Keras
    print("Tokenizando textos...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(processed_texts)
    sequences = tokenizer.texts_to_sequences(processed_texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Preparar etiquetas
    target_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                     'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist',
                     'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
    y = df[target_columns].values
    
    print(f"Vocabulario size: {len(tokenizer.word_index)}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    
    return X, y, tokenizer

def create_model(vocab_size, num_labels):
    model = Sequential([
        # Capa de Embedding mejorada
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        
        # Capa convolucional para capturar n-gramas
        Conv1D(128, 5, activation='relu'),
        
        # LSTM bidireccional para mejor comprensión del contexto
        Bidirectional(LSTM(64, return_sequences=True)),
        
        # Segunda LSTM bidireccional
        Bidirectional(LSTM(32, return_sequences=True)),
        
        # Global Max Pooling
        GlobalMaxPooling1D(),
        
        # Capas densas con dropout más agresivo
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_labels, activation='sigmoid')
    ])
    
    # Optimizador con learning rate ajustado
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def calculate_class_weights(y):
    """
    Calcula pesos de clase para manejar el desbalanceo de manera más robusta
    """
    class_weights = []
    
    # Para cada columna (etiqueta)
    for i in range(y.shape[1]):
        # Obtener los valores únicos y sus pesos
        unique_classes = np.unique(y[:, i])
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y[:, i]
        )
        
        # Crear diccionario de pesos para esta etiqueta
        class_weight_dict = dict(zip(unique_classes, weights))
        class_weights.append(class_weight_dict)
    
    return class_weights

def train_hate_speech_model(df):
    # Preparar datos
    X, y, tokenizer = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calcular pesos de clase
    class_weights = calculate_class_weights(y_train)
    
    # Crear y entrenar modelo
    model = create_model(MAX_WORDS + 1, y.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )
    
    # Modificar el entrenamiento para usar los pesos de clase correctamente
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weights[0]  # Usamos los pesos de la primera clase como ejemplo
    )
    
    # Evaluar modelo
    evaluation = model.evaluate(X_test, y_test)
    print(f"\nTest loss: {evaluation[0]:.4f}")
    print(f"Test accuracy: {evaluation[1]:.4f}")
    print(f"Test AUC: {evaluation[2]:.4f}")
    
    return model, tokenizer, history

# Alternativa: entrenar sin class weights si sigues teniendo problemas
def train_hate_speech_model_no_weights(df):
    # Preparar datos
    X, y, tokenizer = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar modelo
    model = create_model(MAX_WORDS + 1, y.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )
    
    # Entrenar sin class weights
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Evaluar modelo
    evaluation = model.evaluate(X_test, y_test)
    print(f"\nTest loss: {evaluation[0]:.4f}")
    print(f"Test accuracy: {evaluation[1]:.4f}")
    print(f"Test AUC: {evaluation[2]:.4f}")
    
    return model, tokenizer, history

def predict_hate_speech(text, model, tokenizer, threshold=0.5):
    """
    Predicción con umbral ajustable y post-procesamiento
    """
    # Preprocesar texto
    processed_text = preprocess_text(pd.Series([text]))[0]
    sequences = tokenizer.texts_to_sequences([processed_text])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Realizar predicción
    predictions = model.predict(X)
    
    # Aplicar reglas de post-procesamiento
    text_lower = text.lower()
    offensive_words = ['hate', 'bitch', 'fuck', 'kill', 'die']
    if any(word in text_lower for word in offensive_words):
        predictions[0][0] *= 1.2  # Aumentar IsToxic
        predictions[0][1] *= 1.2  # Aumentar IsAbusive
        predictions[0][4] *= 1.2  # Aumentar IsObscene
    
    # Mapear predicciones a etiquetas
    labels = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
              'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist',
              'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
    
    results = {}
    for label, pred in zip(labels, predictions[0]):
        # Aplicar umbral y normalizar
        pred = min(max(pred * 1.2, 0), 1)  # Aumentar sensibilidad general
        results[label] = float(pred)
    
    return results

# Para usar el modelo, puedes elegir entre:
if __name__ == "__main__":
    df = pd.read_csv('data/youtoxic_english_1000.csv')
    
    # Opción 1: Con class weights (si funciona correctamente)
    try:
        print("Intentando entrenar con class weights...")
        model, tokenizer, history = train_hate_speech_model(df)
    except Exception as e:
        print(f"Error al entrenar con class weights: {e}")
        print("\nEntrenando sin class weights...")
        # Opción 2: Sin class weights (como fallback)
        model, tokenizer, history = train_hate_speech_model_no_weights(df)
    
    # Prueba del modelo
    texto_ejemplo = "i hate you son of a bitch"
    predicciones = predict_hate_speech(texto_ejemplo, model, tokenizer)
    print("\nPredicciones para el texto de ejemplo:")
    for label, prob in predicciones.items():
        print(f"{label}: {prob:.4f}")