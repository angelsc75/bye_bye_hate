import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import nltk
import numpy as np


# Descargar recursos necesarios de NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """Evaluar el overfitting comparando la precisión en los conjuntos de entrenamiento y prueba."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Overfitting: {train_accuracy - test_accuracy}')
    
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, vectorizer):
    """Entrenar y evaluar un modelo, y guardar el modelo entrenado."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'--- {model_name} ---')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    evaluate_overfitting(model, X_train, y_train, X_test, y_test)
    joblib.dump(model, f'models/{model_name}.pkl')
    joblib.dump(vectorizer, f'models/{model_name}_vectorizer.pkl')  # Guardar el vectorizador junto con el modelo

def train_models(data):
    """Entrenar varios modelos con ajuste de hiperparámetros usando RandomizedSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned_comments'], data['IsToxic'], test_size=0.2, random_state=42)
    
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)
    
    # Entrenar y evaluar MultinomialNB sin PCA
    nb_model = MultinomialNB()
    train_and_evaluate_model(nb_model, X_train_res, y_train_res, X_test_vec, y_test, 'NaiveBayes', vectorizer)
    
    # Reducción de la dimensionalidad con PCA para otros modelos
    pca = PCA(n_components=0.95)  # Mantener el 95% de la varianza
    X_train_res_pca = pca.fit_transform(X_train_res.toarray())
    X_test_vec_pca = pca.transform(X_test_vec.toarray())
    
    # Guardar PCA
    joblib.dump(pca, 'models/pca.pkl')
    
    # Definir los modelos y sus hiperparámetros
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, penalty='l2'),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'SVC': SVC(probability=True)
    }
    
    # Ajustar los hiperparámetros usando RandomizedSearchCV
    param_distributions = {
        'LogisticRegression': {'C': np.logspace(-3, 3, 7), 'penalty': ['l2']},
        'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]},
        'GradientBoosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]},
        'AdaBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1]},
        'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }
    
    for model_name, model in models.items():
        random_search = RandomizedSearchCV(model, param_distributions[model_name], n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train_res_pca, y_train_res)
        best_model = random_search.best_estimator_
        train_and_evaluate_model(best_model, X_train_res_pca, y_train_res, X_test_vec_pca, y_test, model_name, vectorizer)

# Example usage
if __name__ == "__main__":
    # Load actual dataset
    dataset_path = 'data\youtoxic_english_1000.csv'
    data = pd.read_csv(dataset_path)
    
    # Preprocess text
    data['cleaned_comments'] = data['Text'].apply(preprocess_text)
    
    # Create additional features
    data = create_additional_features(data, 'cleaned_comments')
    
    # Train models
    train_models(data)