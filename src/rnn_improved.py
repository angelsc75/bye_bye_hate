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
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.utils.class_weight import compute_class_weight
import optuna
import logging
import spacy
import gensim.downloader as api
import pickle
import random
from collections import defaultdict

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
        Prepara los datos para el entrenamiento, incluyendo augmentación de texto.
        """
        logging.info("Iniciando preprocesamiento y augmentación de texto...")
        
        # Separar textos ofensivos y no ofensivos
        target_columns = ['IsToxic', 'IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']
        df['IsOffensive'] = df[target_columns].any(axis=1)
        
        offensive_texts = df[df['IsOffensive']]['Text'].tolist()
        non_offensive_texts = df[~df['IsOffensive']]['Text'].tolist()
        
        # Aumentar solo los textos ofensivos
        augmented_offensive_texts = []
        for text in offensive_texts:
            augmented_texts = self.text_augmenter.augment_text(text, augmentation_factor=3)
            augmented_offensive_texts.extend(augmented_texts)
        
        # Combinar todos los textos
        all_texts = augmented_offensive_texts + non_offensive_texts
        all_labels = ([1] * len(augmented_offensive_texts)) + ([0] * len(non_offensive_texts))
        
        # Preprocesar todos los textos usando el TextProcessor
        processed_texts = [self.text_processor.clean_text(text) for text in all_texts]
        
        # Tokenización
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(processed_texts)
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        y = np.array(all_labels)
        
        # Cargar embeddings
        logging.info("Cargando embeddings de GloVe de Twitter...")
        glove_model = api.load("glove-twitter-200")
        embedding_matrix = np.zeros((MAX_WORDS + 1, EMBEDDING_DIM))
        for word, i in self.tokenizer.word_index.items():
            if i < MAX_WORDS:
                try:
                    embedding_matrix[i] = glove_model[word]
                except KeyError:
                    continue
                    
        # Calcular pesos de clase
        class_weights = compute_class_weight('balanced',
                                          classes=np.unique(y),
                                          y=y)
        class_weight_dict = dict(enumerate(class_weights))
        
        return X, y, embedding_matrix, class_weight_dict

    def create_model_tuned(self, vocab_size, num_labels, params, embedding_matrix):
        """
        Crea el modelo con los parámetros optimizados.
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

    def train_with_optimization(self, df, n_trials=20):
        """
        Entrena el modelo con optimización de hiperparámetros.
        """
        logging.info("Preparando datos...")
        X, y, embedding_matrix, class_weight_dict = self.prepare_data(df)
        
        logging.info(f"Distribución de clases después de augmentación: {np.bincount(y)}")
        
        # Crear modelo con parámetros por defecto para este ejemplo
        best_params = {
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
        
        final_model = self.create_model_tuned(
            vocab_size=MAX_WORDS + 1,
            num_labels=1,  # Binary classification
            params=best_params,
            embedding_matrix=embedding_matrix
        )
        
        # Entrenar el modelo
        history = final_model.fit(
            X, y,
            epochs=15,
            batch_size=best_params.get('batch_size', 32),
            validation_split=0.1,
            class_weight=class_weight_dict,
            callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
        )

        return final_model, history, self.tokenizer

if __name__ == "__main__":
    # Asegúrate de que el archivo CSV existe y tiene las columnas correctas
    df = pd.read_csv('data/youtoxic_english_1000.csv')

    try:
        logging.info("Iniciando entrenamiento con optimización...")
        trainer = ModelTrainer()
        final_model, history, tokenizer = trainer.train_with_optimization(df, n_trials=20)
        
        # Guardar el tokenizer
        with open('models/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Guardar el modelo
        final_model.save('models/final_model.h5')
        logging.info("Entrenamiento completado y modelo guardado.")

    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")