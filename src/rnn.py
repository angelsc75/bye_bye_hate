import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Importaciones de Keras corregidas
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, 
    LSTM, 
    Dense, 
    Dropout, 
    Bidirectional, 
    GlobalMaxPooling1D, 
    Conv1D,
    Input  # Añadido aquí correctamente
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Importaciones de NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk

# Otras importaciones
import optuna
import logging
import spacy
import gensim.downloader as api
import pickle
import random
from collections import defaultdict
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature


# Configuración inicial
logging.basicConfig(level=logging.INFO)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Configuración de parámetros base
MAX_WORDS = 15000
MAX_LEN = 128
EMBEDDING_DIM = 200

# Diccionario de palabras ofensivas y sus sinónimos
OFFENSIVE_SYNONYMS = {
    'hate': ['despise', 'loathe', 'detest', 'abhor'],
    'stupid': ['idiotic', 'moronic', 'brainless', 'foolish'],
    'idiot': ['moron', 'imbecile', 'fool', 'dunce'],
    'kill': ['murder', 'slay', 'eliminate', 'destroy'],
    'ugly': ['hideous', 'grotesque', 'repulsive', 'disgusting'],
    'fat': ['obese', 'overweight', 'huge', 'massive'],
    'die': ['perish', 'expire', 'deceased', 'dead'],
    'racist': ['bigot', 'discriminatory', 'prejudiced', 'biased'],
    'trash': ['garbage', 'rubbish', 'waste', 'junk'],
}

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) - {'no', 'not', 'hate', 'against', 'racist', 'abuse', 'toxic'}
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
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
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        return ' '.join(words)

    def preprocess_text(self, text_series):
        """
        Preprocesa una serie de textos.
        """
        return text_series.apply(self.clean_text)

class TextAugmenter:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.offensive_patterns = self._compile_offensive_patterns()
        self.text_processor = TextProcessor()

    def _get_wordnet_pos(self, word, tag):
        """Convierte el tag POS de NLTK al formato de WordNet."""
        tag_dict = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }
        return tag_dict.get(tag[0], wordnet.NOUN)

    def _compile_offensive_patterns(self):
        """Compila patrones regex para palabras ofensivas."""
        patterns = {}
        for word in OFFENSIVE_SYNONYMS:
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
            patterns[word] = pattern
        return patterns

    def _get_synonyms(self, word):
        """Obtiene sinónimos de WordNet y del diccionario personalizado."""
        synonyms = set()
        
        # Buscar en el diccionario personalizado
        if word.lower() in OFFENSIVE_SYNONYMS:
            synonyms.update(OFFENSIVE_SYNONYMS[word.lower()])
        
        # Buscar en WordNet
        pos_tagged = nltk.pos_tag([word])
        pos = self._get_wordnet_pos(word, pos_tagged[0][1])
        
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                if lemma.name() != word and "_" not in lemma.name():
                    synonyms.add(lemma.name())
        
        return list(synonyms)

    def augment_text(self, text, augmentation_factor=2):
        """
        Aumenta el texto reemplazando palabras ofensivas con sus sinónimos.
        """
        words = text.split()
        augmented_texts = [text]  # Incluye el texto original
        
        for _ in range(augmentation_factor - 1):
            new_words = words.copy()
            replacements_made = False
            
            for i, word in enumerate(words):
                # Verifica si la palabra está en nuestro diccionario de palabras ofensivas
                word_lower = word.lower()
                if word_lower in OFFENSIVE_SYNONYMS:
                    synonyms = OFFENSIVE_SYNONYMS[word_lower]
                    if synonyms:
                        new_words[i] = random.choice(synonyms)
                        replacements_made = True
                        
            if replacements_made:
                augmented_texts.append(" ".join(new_words))
            
        return augmented_texts

class ModelTrainer:
    def __init__(self):
        self.tokenizer = None
        self.text_augmenter = TextAugmenter()
        self.text_processor = TextProcessor()
        
    def prepare_data(self, df):
        """
        Prepara los datos para el entrenamiento, incluyendo augmentación de texto
        y asegurando la forma correcta de los datos.
        """
        logging.info("Iniciando preprocesamiento y augmentación de texto...")
        
        # Verificar y limpiar valores nulos
        df['Text'] = df['Text'].fillna('')
        
        # Separar textos ofensivos y no ofensivos
        target_columns = ['IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']
        df['IsOffensive'] = df[target_columns].any(axis=1)
        
        # Convertir a lista y asegurar que son strings
        offensive_texts = df[df['IsOffensive']]['Text'].astype(str).tolist()
        non_offensive_texts = df[~df['IsOffensive']]['Text'].astype(str).tolist()
        
        # Aumentar solo los textos ofensivos
        logging.info(f"Augmentando {len(offensive_texts)} textos ofensivos...")
        augmented_offensive_texts = []
        for text in offensive_texts:
            augmented_texts = self.text_augmenter.augment_text(text, augmentation_factor=3)
            augmented_offensive_texts.extend(augmented_texts)
        
        # Combinar todos los textos
        all_texts = augmented_offensive_texts + non_offensive_texts
        all_labels = np.array([1] * len(augmented_offensive_texts) + [0] * len(non_offensive_texts))
        
        # Preprocesar todos los textos
        logging.info("Preprocesando textos...")
        processed_texts = [self.text_processor.clean_text(text) for text in all_texts]
        
        # Inicializar y ajustar el tokenizer si no existe
        if self.tokenizer is None:
            logging.info("Inicializando tokenizer...")
            self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(processed_texts)
        
        # Convertir textos a secuencias
        logging.info("Convirtiendo textos a secuencias...")
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        
        # Padding de secuencias
        logging.info(f"Aplicando padding a las secuencias (max_len={MAX_LEN})...")
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Verificar y ajustar las dimensiones
        if len(X.shape) != 2:
            raise ValueError(f"Forma incorrecta de los datos de entrada. Se esperaba 2D, pero se obtuvo {len(X.shape)}D")
        
        if X.shape[1] != MAX_LEN:
            raise ValueError(f"Longitud de secuencia incorrecta. Se esperaba {MAX_LEN}, pero se obtuvo {X.shape[1]}")
        
        # Verificar que las etiquetas sean un array 1D
        y = all_labels.reshape(-1)
        
        logging.info(f"Preparación de datos completada. Shape de X: {X.shape}, Shape de y: {y.shape}")
        
        return X, y

    def load_embeddings(self):
        """Carga los embeddings de GloVe."""
        logging.info("Cargando embeddings de GloVe de Twitter...")
        glove_model = api.load("glove-twitter-200")
        embedding_matrix = np.zeros((MAX_WORDS + 1, EMBEDDING_DIM))
        
        for word, i in self.tokenizer.word_index.items():
            if i < MAX_WORDS:
                try:
                    embedding_matrix[i] = glove_model[word]
                except KeyError:
                    continue
                    
        return embedding_matrix

    def create_model(self, params, embedding_matrix):
        """
        Crea el modelo con los parámetros dados y forma de entrada explícita.
        """
        model = Sequential([
            # Definir explícitamente la forma de entrada
            Input(shape=(MAX_LEN,)),
            Embedding(MAX_WORDS + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False),
            Conv1D(params['conv_filters'], 5, activation='relu'),
            Bidirectional(LSTM(params['lstm_units_1'], return_sequences=True)),
            Bidirectional(LSTM(params['lstm_units_2'], return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(params['dense_units_1'], activation='relu'),
            Dropout(params['dropout_1']),
            Dense(params['dense_units_2'], activation='relu'),
            Dropout(params['dropout_2']),
            Dense(1, activation='sigmoid')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        return model

    def train_with_cv(self, df, n_splits=5):
        """
        Entrena el modelo usando validación cruzada y registra los resultados con MLflow.
        """
        mlflow.set_experiment("toxic_text_classification")
        
        # Preparar datos
        X, y = self.prepare_data(df)
        embedding_matrix = self.load_embeddings()
        
        # Parámetros del modelo
        params = {
            'conv_filters': 128,
            'lstm_units_1': 64,
            'lstm_units_2': 32,
            'dense_units_1': 128,
            'dense_units_2': 64,
            'dropout_1': 0.5,
            'dropout_2': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 15
        }
        
        # Inicializar K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        # Deshabilitar el autolog de TensorFlow para evitar warnings redundantes
        mlflow.tensorflow.autolog(log_models=False)
        
        with mlflow.start_run() as run:
            # Registrar parámetros
            mlflow.log_params(params)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logging.info(f"Training fold {fold + 1}/{n_splits}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Calcular pesos de clase para este fold
                class_weights = compute_class_weight('balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
                class_weight_dict = dict(enumerate(class_weights))
                
                # Crear y entrenar el modelo
                model = self.create_model(params, embedding_matrix)
                
                history = model.fit(
                    X_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=(X_val, y_val),
                    class_weight=class_weight_dict,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
                )
                
                # Evaluar el modelo
                scores = model.evaluate(X_val, y_val)
                metrics = dict(zip(model.metrics_names, scores))
                fold_metrics.append(metrics)
                
                # Registrar métricas para este fold
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"fold_{fold+1}_{metric_name}", value)
                
                # Guardar las curvas de aprendizaje para este fold
                for metric in history.history:
                    mlflow.log_metric(f"fold_{fold+1}_training_{metric}", history.history[metric][-1])
            
            # Calcular y registrar métricas promedio
            avg_metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in fold_metrics[0].keys()
            }
            
            for metric_name, value in avg_metrics.items():
                
                # Calcular y registrar la diferencia entre accuracies de entrenamiento y validación
                if 'val_accuracy' in history.history and 'accuracy' in history.history:
                    training_accuracy = history.history['accuracy'][-1]
                    validation_accuracy = history.history['val_accuracy'][-1]
                    accuracy_difference = training_accuracy - validation_accuracy
                    mlflow.log_metric('accuracy_difference', accuracy_difference)

                # Calcular y registrar la métrica de overfitting
                if 'val_loss' in history.history and 'loss' in history.history:
                    training_loss = history.history['loss'][-1]
                    validation_loss = history.history['val_loss'][-1]
                    if training_loss > 0:  # Evitar división por cero
                        overfitting_ratio = validation_loss / training_loss
                        mlflow.log_metric('overfitting_ratio', overfitting_ratio)
                mlflow.log_metric(f"average_{metric_name}", value)
            
            # Guardar el mejor modelo (usando el último fold como referencia)
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.keras.log_model(model, "model", signature=signature)
            
            # Guardar el tokenizer
            with open('models/tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact('models/tokenizer.pickle')
            
        return model, self.tokenizer, avg_metrics

if __name__ == "__main__":
    # Configurar MLflow
    mlflow.tensorflow.autolog()
    
    # Cargar datos
    df = pd.read_csv('data/youtoxic_english_1000.csv')
    
    try:
        logging.info("Iniciando entrenamiento con validación cruzada...")
        trainer = ModelTrainer()
        model, tokenizer, metrics = trainer.train_with_cv(df, n_splits=5)
        
        logging.info(f"Métricas promedio del modelo:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")

# Guardar el modelo al final del entrenamiento
model.save('models/final_model.h5')
