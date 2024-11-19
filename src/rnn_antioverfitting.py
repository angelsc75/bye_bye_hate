# Importación de librerías necesarias
# --- Librerías de procesamiento de datos ---
import pandas as pd  # Para manipulación de datos tabulares
import numpy as np   # Para operaciones numéricas
import re           # Para expresiones regulares

# --- Librerías de Machine Learning ---
from sklearn.model_selection import StratifiedKFold, train_test_split  # Para división de datos y validación cruzada
from sklearn.preprocessing import LabelEncoder  # Para codificación de etiquetas
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # Para métricas de evaluación
from sklearn.utils.class_weight import compute_class_weight  # Para manejar desbalance de clases

# --- Librerías de Deep Learning (TensorFlow/Keras) ---
from tensorflow.keras.preprocessing.text import Tokenizer  # Para tokenización de texto
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Para hacer secuencias de igual longitud
from tensorflow.keras.models import Sequential  # Para crear modelo secuencial
# Capas del modelo neural
from tensorflow.keras.layers import (
    Embedding,  # Para embeddings de palabras
    LSTM,       # Capa LSTM para procesamiento secuencial
    Dense,      # Capa completamente conectada
    Dropout,    # Para prevenir overfitting
    Bidirectional,  # Para LSTM bidireccional
    GlobalMaxPooling1D,  # Para reducir dimensionalidad
    Conv1D,     # Para convolución 1D
    Input,      # Capa de entrada
    BatchNormalization,  # Para normalizar los datos
    SpatialDropout1D    # Dropout espacial para embeddings
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Callbacks para entreamiento
from tensorflow.keras.regularizers import l2  # Para regularización L2
import tensorflow as tf

# --- Librerías de procesamiento de lenguaje natural ---
from nltk.tokenize import word_tokenize  # Para tokenización
from nltk.corpus import stopwords, wordnet  # Para stopwords y sinónimos
from nltk.stem import WordNetLemmatizer  # Para lematización
import nltk

# --- Otras librerías útiles ---
import optuna  # Para optimización de hiperparámetros
import logging  # Para registro de eventos
import spacy   # Para procesamiento de lenguaje natural
import gensim.downloader as api  # Para cargar word embeddings
import pickle  # Para guardar/cargar objetos
import random  # Para operaciones aleatorias
from collections import defaultdict  # Para diccionarios con valor por defecto
import mlflow  # Para tracking de experimentos
import mlflow.keras
from mlflow.models.signature import infer_signature
from googletrans import Translator  # Para traducción

# Configuración inicial
logging.basicConfig(level=logging.INFO)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Parámetros globales del modelo
MAX_WORDS = 15000    # Máximo número de palabras en el vocabulario
MAX_LEN = 128       # Longitud máxima de las secuencias
EMBEDDING_DIM = 200  # Dimensión de los embeddings

# Diccionario expandido de palabras ofensivas y sus sinónimos
# Esto ayuda a aumentar el dataset y mejorar la detección
OFFENSIVE_SYNONYMS = {
    'hate': ['despise', 'loathe', 'detest', 'abhor', 'resent', 'dislike'],
    'stupid': ['idiotic', 'moronic', 'brainless', 'foolish', 'dense', 'dull'],
    'idiot': ['moron', 'imbecile', 'fool', 'dunce', 'dimwit', 'dolt'],
    'kill': ['murder', 'slay', 'eliminate', 'destroy', 'annihilate', 'execute'],
    'ugly': ['hideous', 'grotesque', 'repulsive', 'disgusting', 'unsightly', 'homely'],
    'fat': ['obese', 'overweight', 'huge', 'massive', 'bulky', 'heavy'],
    'die': ['perish', 'expire', 'deceased', 'dead', 'gone', 'departed'],
    'racist': ['bigot', 'discriminatory', 'prejudiced', 'biased', 'xenophobic', 'intolerant'],
    'trash': ['garbage', 'rubbish', 'waste', 'junk', 'refuse', 'debris'],
}

class TextProcessor:
    """
    Clase para procesar y limpiar texto
    """
    def __init__(self):
        # Inicializar stopwords excluyendo palabras importantes para la detección
        self.stop_words = set(stopwords.words('english')) - {'no', 'not', 'hate', 'against', 'racist', 'abuse', 'toxic'}
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Limpia y preprocesa el texto
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Preservar palabras compuestas ofensivas
        text = text.replace('son of a bitch', 'sonofabitch')
        text = text.replace('f u c k', 'fuck')
        text = text.replace('b i t c h', 'bitch')
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Preservar caracteres especiales relevantes
        text = re.sub(r'[^a-zA-Z\s!?*#@$]', '', text)
        
        # Eliminar espacios extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lematización y eliminación de stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

class TextAugmenter:
    """
    Clase para aumentar datos de texto usando diferentes técnicas
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.offensive_patterns = self._compile_offensive_patterns()
        self.text_processor = TextProcessor()
        self.translator = Translator()

    def _get_wordnet_pos(self, word, tag):
        tag_dict = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }
        return tag_dict.get(tag[0], wordnet.NOUN)

    def _compile_offensive_patterns(self):
        patterns = {}
        for word in OFFENSIVE_SYNONYMS:
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
            patterns[word] = pattern
        return patterns

    def _get_synonyms(self, word):
        synonyms = set()
        
        if word.lower() in OFFENSIVE_SYNONYMS:
            synonyms.update(OFFENSIVE_SYNONYMS[word.lower()])
        
        pos_tagged = nltk.pos_tag([word])
        pos = self._get_wordnet_pos(word, pos_tagged[0][1])
        
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                if lemma.name() != word and "_" not in lemma.name():
                    synonyms.add(lemma.name())
        
        return list(synonyms)

    def _back_translate(self, text, intermediate_lang='es'):
        """Realiza back translation usando un idioma intermedio"""
        try:
            # Traducir al idioma intermedio
            intermediate = self.translator.translate(text, dest=intermediate_lang).text
            # Traducir de vuelta al inglés
            back_translated = self.translator.translate(intermediate, dest='en').text
            return back_translated
        except:
            return text

    def _random_deletion(self, text, p=0.1):
        """Elimina palabras aleatoriamente con probabilidad p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)

    def augment_text(self, text, augmentation_factor=2):
        """
        Aumenta el texto usando múltiples técnicas como:
        - Sustitución de sinónimos
        - Back translation
        - Eliminación aleatoria de palabras
        """
        augmented_texts = [text]
        
        for _ in range(augmentation_factor - 1):
            new_text = text
            # Aplicar técnicas de aumentación aleatoriamente
            if random.random() < 0.7:
                words = new_text.split()
                new_words = words.copy()
                for i, word in enumerate(words):
                    if word.lower() in OFFENSIVE_SYNONYMS and random.random() < 0.5:
                        synonyms = OFFENSIVE_SYNONYMS[word.lower()]
                        if synonyms:
                            new_words[i] = random.choice(synonyms)
                new_text = ' '.join(new_words)
            
            if random.random() < 0.3:
                new_text = self._back_translate(new_text)
            
            if random.random() < 0.2:
                new_text = self._random_deletion(new_text)
            
            if new_text != text:
                augmented_texts.append(new_text)
        
        return augmented_texts

class ModelTrainer:
    """
    Clase principal para entrenar el modelo
    """
    def __init__(self):
        self.tokenizer = None
        self.text_augmenter = TextAugmenter()
        self.text_processor = TextProcessor()
        
    def prepare_data(self, df):
        """
        Prepara los datos para el entrenamiento:
        1. Limpia y procesa el texto
        2. Aumenta los datos de la clase minoritaria
        3. Tokeniza y padea las secuencias
        """
        logging.info("Iniciando preprocesamiento y augmentación de texto...")
        
        df['Text'] = df['Text'].fillna('')
        
        target_columns = ['IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']
        df['IsOffensive'] = df[target_columns].any(axis=1)
        
        offensive_texts = df[df['IsOffensive']]['Text'].astype(str).tolist()
        non_offensive_texts = df[~df['IsOffensive']]['Text'].astype(str).tolist()
        
        logging.info(f"Augmentando {len(offensive_texts)} textos ofensivos...")
        self.augmented_offensive_texts = []  # Guardar como atributo de la clase
        for text in offensive_texts:
            augmented_texts = self.text_augmenter.augment_text(text, augmentation_factor=3)
            self.augmented_offensive_texts.extend(augmented_texts)
        
        all_texts = self.augmented_offensive_texts + non_offensive_texts
        all_labels = np.array([1] * len(self.augmented_offensive_texts) + [0] * len(non_offensive_texts))
        
        processed_texts = [self.text_processor.clean_text(text) for text in all_texts]
        
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(processed_texts)
        
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        if len(X.shape) != 2:
            raise ValueError(f"Forma incorrecta de los datos de entrada. Se esperaba 2D, pero se obtuvo {len(X.shape)}D")
        
        if X.shape[1] != MAX_LEN:
            raise ValueError(f"Longitud de secuencia incorrecta. Se esperaba {MAX_LEN}, pero se obtuvo {X.shape[1]}")
        
        y = all_labels.reshape(-1)
        
        return X, y

    def load_embeddings(self):
        logging.info("Cargando embeddings de GloVe...")
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
        Crea el modelo neural con la siguiente arquitectura:
        1. Capa de Embedding
        2. Capa Convolucional 1D
        3. LSTM Bidireccional
        4. Capas Dense
        Incluye regularización y normalización para prevenir overfitting
        """
           
    # Definir la métrica F1 personalizada
        class BinaryF1Score(tf.keras.metrics.Metric):
            def __init__(self, name='f1', **kwargs):
                super().__init__(name=name, **kwargs)
                self.precision = tf.keras.metrics.Precision()
                self.recall = tf.keras.metrics.Recall()

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_pred = tf.cast(y_pred > 0.5, tf.float32)
                self.precision.update_state(y_true, y_pred, sample_weight)
                self.recall.update_state(y_true, y_pred, sample_weight)

            def result(self):
                p = self.precision.result()
                r = self.recall.result()
                return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

            def reset_state(self):
                self.precision.reset_state()
                self.recall.reset_state()

        # Definir el learning rate schedule
        initial_learning_rate = params['learning_rate']
        decay_steps = 1000
        decay_rate = 0.9
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        model = Sequential([

            # Entrada de datos: secuencias de texto ya procesadas y con padding aplicado.
            Input(shape=(MAX_LEN,)),  # El tamaño de entrada es la longitud máxima de las secuencias (MAX_LEN).

            # Embedding layer: convierte palabras en vectores densos usando embeddings preentrenados.
            Embedding(
                MAX_WORDS + 1,  # Número máximo de palabras en el vocabulario más un token para OOV (<OOV>).
                EMBEDDING_DIM,  # Dimensión del embedding (por ejemplo, 200).
                weights=[embedding_matrix],  # Usamos embeddings preentrenados de GloVe.
                trainable=False  # No se entrenan los embeddings; se usan tal cual.
            ),

            # SpatialDropout1D: desactiva aleatoriamente características en el embedding para regularización.
            SpatialDropout1D(0.3),  # Previene el sobreajuste desactivando características en cada paso.

            # BatchNormalization: normaliza los valores a lo largo del batch para estabilizar y acelerar el entrenamiento.
            BatchNormalization(),

            # Conv1D: una capa convolucional para extraer características locales del texto.
            Conv1D(
                params['conv_filters'],  # Número de filtros (parámetro ajustable).
                5,  # Tamaño del kernel (analiza 5 palabras consecutivas).
                activation='relu',  # Activación ReLU para no linealidad.
                kernel_regularizer=l2(0.02),  # Regularización L2 para evitar sobreajuste.
                bias_regularizer=l2(0.02)  # Regularización para los sesgos también.
            ),

            # BatchNormalization: vuelve a normalizar las características después de la convolución.
            BatchNormalization(),

            # Dropout: desactiva nodos aleatoriamente para reducir el sobreajuste.
            Dropout(0.4),  # El 40% de las neuronas se desactivan durante cada paso.

            # Bidirectional LSTM: analiza las secuencias en ambas direcciones (hacia adelante y hacia atrás).
            Bidirectional(
                LSTM(
                    params['lstm_units_1'] // 2,  # Número de unidades LSTM dividido entre 2 (una mitad para cada dirección).
                    return_sequences=True,  # Mantiene todas las salidas para procesarlas más adelante.
                    kernel_regularizer=l2(0.02),  # Regularización L2 para los pesos.
                    recurrent_regularizer=l2(0.02),  # Regularización en las conexiones recurrentes.
                    bias_regularizer=l2(0.02)  # Regularización en los sesgos.
                )
            ),

            # BatchNormalization: normaliza las salidas de la capa LSTM bidireccional.
            BatchNormalization(),

            # Dropout: añade otro nivel de regularización para la capa recurrente.
            Dropout(0.4),  # El 40% de las neuronas se desactivan.

            # GlobalMaxPooling1D: reduce las dimensiones extrayendo los valores máximos de las características.
            GlobalMaxPooling1D(),  # Toma el valor más alto por característica para capturar la información más relevante.

            # Primera capa densa: combina las características extraídas de forma no lineal.
            Dense(
                params['dense_units_1'] // 2,  # Número de neuronas en la capa densa (ajustable).
                activation='relu',  # Activación ReLU.
                kernel_regularizer=l2(0.02)  # Regularización L2 para los pesos.
            ),

            # Dropout: más regularización para evitar que la capa densa memorice.
            Dropout(0.5),  # El 50% de las neuronas se desactivan.

            # Capa de salida: produce una probabilidad binaria para la clasificación.
            Dense(
                1,  # Una sola salida (0 o 1).
                activation='sigmoid'  # Activación sigmoidal para una salida entre 0 y 1 (probabilidad).
            )
        ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    BinaryF1Score(name='f1')]
        )
        return model

    def train_with_cv(self, df, n_splits=5):
        """
        Entrena el modelo usando validación cruzada y:
        1. Registra métricas con MLflow
        2. Implementa early stopping
        3. Reduce learning rate cuando sea necesario
        4. Guarda el mejor modelo
        """
        X, y = self.prepare_data(df)
        embedding_matrix = self.load_embeddings()
        
        params = {
            'conv_filters': 32,
            'lstm_units_1': 16,
            'dense_units_1': 32,
            'dropout_1': 0.5,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 20
        }
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                logging.info(f"Training fold {fold + 1}/{n_splits}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                class_weights = compute_class_weight('balanced',
                                                classes=np.unique(y_train),
                                                y=y_train)
                class_weight_dict = dict(enumerate(class_weights))
                
                model = self.create_model(params, embedding_matrix)
                
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        min_delta=0.001,
                        mode='min'
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=3,
                        min_lr=1e-6,
                        mode='min'
                    )
                ]
                
                history = model.fit(
                    X_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=(X_val, y_val),
                    validation_freq=1,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    shuffle=True
                )
                
                # Log métricas manualmente
                scores = model.evaluate(X_val, y_val, verbose=0)
                metrics = dict(zip(model.metrics_names, scores))
                fold_metrics.append(metrics)
                
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"fold_{fold+1}_{metric_name}", value)
                
                for metric in history.history:
                    mlflow.log_metric(f"fold_{fold+1}_training_{metric}", 
                                    history.history[metric][-1])
            
            avg_metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                
    for metric in fold_metrics[0].keys()
            }
            
            for metric_name, value in avg_metrics.items():
                # Calcular métricas de overfitting
                if 'val_accuracy' in history.history and 'accuracy' in history.history:
                    training_accuracy = history.history['accuracy'][-1]
                    validation_accuracy = history.history['val_accuracy'][-1]
                    accuracy_difference = abs(training_accuracy - validation_accuracy)
                    mlflow.log_metric('accuracy_difference', accuracy_difference)
                    
                    # Log de alerta si hay señales de overfitting
                    if accuracy_difference > 0.05:  # threshold arbitrario del 10%
                        logging.warning(f"Posible overfitting detectado: diferencia de accuracy = {accuracy_difference:.4f}")

                if 'val_loss' in history.history and 'loss' in history.history:
                    training_loss = history.history['loss'][-1]
                    validation_loss = history.history['val_loss'][-1]
                    if training_loss > 0:
                        overfitting_ratio = validation_loss / training_loss
                        mlflow.log_metric('overfitting_ratio', overfitting_ratio)
                        
                        if overfitting_ratio > 1.2:  # threshold arbitrario del 20%
                            logging.warning(f"Posible overfitting detectado: ratio de pérdida = {overfitting_ratio:.4f}")
                
                mlflow.log_metric(f"average_{metric_name}", value)
            
            # Guardar el mejor modelo
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.keras.log_model(model, "model", signature=signature)
            
            # Guardar el tokenizer
            with open('models/tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact('models/tokenizer.pickle')
            
            # Guardar información sobre la augmentación de datos
        augmentation_info = {
            "original_offensive_samples": len(df[df['IsOffensive']]),
            "augmented_offensive_samples": len(self.augmented_offensive_texts),
            "augmentation_factor": len(self.augmented_offensive_texts) / len(df[df['IsOffensive']]),
            "total_samples": len(X)
        }
        mlflow.log_params(augmentation_info)
            
        return model, self.tokenizer, avg_metrics

def evaluate_model_performance(model, X_test, y_test):
    """
    Evalúa el modelo usando múltiples métricas:
    - Precisión
    - Recall
    - F1-score
    - AUC-ROC
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calcular métricas básicas
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_binary, average='binary')
    auc = roc_auc_score(y_test, y_pred)
    
    # Crear diccionario de métricas
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    return metrics

if __name__ == "__main__":
    # Configurar logging detallado
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Cargar datos
        df = pd.read_csv('data/youtoxic_english_1000.csv')
        logging.info(f"Datos cargados: {len(df)} muestras")
        
        # Separar conjunto de prueba estratificado
        target_columns = ['IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']
        df['IsOffensive'] = df[target_columns].any(axis=1)
        
        train_df, test_df = train_test_split(
            df, 
            test_size=0.1, 
            stratify=df['IsOffensive'],
            random_state=42
        )
        
        # Configurar MLflow - Mover la configuración aquí y desactivar el autolog
        mlflow.set_experiment("toxic_text_classification")
        # Desactivar el autolog global
        mlflow.tensorflow.autolog(log_models=False, disable=True)
        
        logging.info("Iniciando entrenamiento con validación cruzada...")
        trainer = ModelTrainer()
        
        # Entrenar modelo
        model, tokenizer, metrics = trainer.train_with_cv(train_df, n_splits=5)
        
        # Preparar datos de prueba
        X_test, y_test = trainer.prepare_data(test_df)
        
        # Evaluar en conjunto de prueba
        test_metrics = evaluate_model_performance(model, X_test, y_test)
        
        # Registrar resultados
        logging.info("\nMétricas promedio del modelo en validación cruzada:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
            
        logging.info("\nMétricas en conjunto de prueba:")
        for metric_name, value in test_metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
        
        # Guardar el modelo final
        model.save('models/final_model.h5')
        logging.info("Modelo guardado exitosamente en 'models/final_model.h5'")
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        raise e