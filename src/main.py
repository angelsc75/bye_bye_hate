import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer  # Cambiado a TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2  # Para selección de características
import optuna
import joblib
import nltk
import warnings
warnings.filterwarnings('ignore')

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

def create_additional_features(df, text_column):
    """Crear características adicionales como la longitud de los comentarios y la frecuencia de ciertas palabras."""
    df['comment_length'] = df[text_column].apply(lambda x: len(x.split()))
    df['hate_frequency'] = df[text_column].apply(lambda x: x.lower().split().count('hate'))
    return df

def evaluate_overfitting(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluar detalladamente el overfitting del modelo."""
    # Métricas en conjunto de entrenamiento
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Métricas en conjunto de prueba
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # Calcular diferencia (métrica de overfitting)
    overfitting_gap = train_accuracy - test_accuracy
    
    print(f"\n{'='*20} Métricas de Overfitting para {model_name} {'='*20}")
    print(f"Accuracy en Training: {train_accuracy:.4f}")
    print(f"Accuracy en Test: {test_accuracy:.4f}")
    print(f"Diferencia (Overfitting Gap): {overfitting_gap:.4f}")
    
    # Interpretar el nivel de overfitting
    if overfitting_gap < 0.05:
        print("Nivel de Overfitting: Bajo ✅")
    elif overfitting_gap < 0.1:
        print("Nivel de Overfitting: Moderado ⚠️")
    else:
        print("Nivel de Overfitting: Alto ❌")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting_gap': overfitting_gap
    }

def evaluate_model_cv(model, X, y, cv=5):
    """Evaluar el modelo usando validación cruzada."""
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    return cv_scores.mean()

def optimize_logistic_regression(trial, X_train, y_train):
    """Función objetivo para optimización con Optuna - Regresión Logística."""
    C = trial.suggest_loguniform('C', 1e-5, 100)
    max_iter = trial.suggest_int('max_iter', 100, 2000)
    
    model = LogisticRegression(C=C, max_iter=max_iter)
    return evaluate_model_cv(model, X_train, y_train)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, vectorizer):
    """Entrenar y evaluar un modelo, y guardar el modelo entrenado."""
    print(f"\nEntrenando {model_name}...")
    model.fit(X_train, y_train)
    
    # Evaluar overfitting
    overfitting_metrics = evaluate_overfitting(model, X_train, y_train, X_test, y_test, model_name)
    
    # Mostrar reporte de clasificación
    y_pred = model.predict(X_test)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo y vectorizador
    joblib.dump(model, f'models/{model_name}.pkl')
    joblib.dump(vectorizer, f'models/{model_name}_vectorizer.pkl')
    
    return overfitting_metrics

def train_models(data):
    """Entrenar modelos con técnicas de reducción de overfitting."""
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_comments'], data['IsToxic'], test_size=0.2, random_state=42
    )
    
    # 1. Usar TF-IDF con límites de características
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,  # Limitar número de características
        min_df=2,          # Ignorar términos que aparecen en menos de 2 documentos
        max_df=0.95        # Ignorar términos que aparecen en más del 95% de los documentos
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 2. Selección de características
    selector = SelectKBest(chi2, k=3000)
    X_train_selected = selector.fit_transform(X_train_vec, y_train)
    X_test_selected = selector.transform(X_test_vec)
    
    # 3. Aplicar SMOTE de manera más conservadora
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_res, y_train_res = smote.fit_resample(X_train_selected, y_train)
    
    # 4. Configurar modelos con regularización más fuerte
    models = {
        'LogisticRegression': LogisticRegression(
            C=0.1,                # Aumentar regularización
            max_iter=2000,
            penalty='l2',
            class_weight='balanced'  # Manejar desbalance de clases
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=8,          # Reducir profundidad máxima
            min_samples_split=10,  # Aumentar muestras mínimas para split
            min_samples_leaf=5,    # Aumentar muestras mínimas en hojas
            max_features='sqrt',   # Limitar características por split
            random_state=42,
            class_weight='balanced'  # Manejar desbalance de clases
        ),
        'NaiveBayes': MultinomialNB(
            alpha=1.0  # Aumentar suavizado de Laplace
        )
    }
    
    # 5. Grid Search con parámetros más conservadores para Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 8, 10],
        'min_samples_split': [8, 10, 12],
        'min_samples_leaf': [4, 5, 6]
    }
    
    rf_grid = GridSearchCV(
        models['RandomForest'],
        rf_param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='balanced_accuracy'  # Usar métrica balanceada
    )
    
    # Entrenar y evaluar modelos
    overfitting_results = {}
    
    for name, model in models.items():
        if name == 'RandomForest':
            print("\nOptimizando Random Forest con GridSearchCV...")
            rf_grid.fit(X_train_res, y_train_res)
            model = rf_grid.best_estimator_
            print(f"Mejores parámetros: {rf_grid.best_params_}")
        
        overfitting_results[name] = train_and_evaluate_model(
            model, X_train_res, y_train_res, X_test_selected, y_test, name, vectorizer
        )
    
    # Guardar el selector de características
    joblib.dump(selector, 'models/feature_selector.pkl')
    
    # Mostrar comparativa final
    print("\n" + "="*50)
    print("COMPARATIVA DE OVERFITTING ENTRE MODELOS")
    print("="*50)
    for model_name, metrics in overfitting_results.items():
        print(f"\n{model_name}:")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Overfitting Gap: {metrics['overfitting_gap']:.4f}")

if __name__ == "__main__":
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Load and preprocess data
    dataset_path = 'data/youtoxic_english_1000.csv'
    data = pd.read_csv(dataset_path)
    data['cleaned_comments'] = data['Text'].apply(preprocess_text)
    data = create_additional_features(data, 'cleaned_comments')
    
    # Train models
    train_models(data)