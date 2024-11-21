# Detector de Mensajes de Odio en YouTube

## 📝 Descripción
Sistema de detección y análisis de mensajes de odio en comentarios de YouTube utilizando Deep Learning y Natural Language Processing. Implementado con Redes Neuronales Convolucionales  y Streamlit.

## 🛠️ Tecnologías Principales
- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Deep Learning**: TensorFlow 2.15.0
- **Base de Datos**: MongoDB 4.6.1 , Mongo-Express 1.0.2
- **NLP**: NLTK 3.8.1, Spacy 3.7.2
- **API**: YouTube Data API v3

## 📋 Requisitos del Sistema
- Python 3.8 o superior
- MongoDB instalado y en ejecución
- Memoria RAM: 8GB mínimo recomendado
- Espacio en disco: 2GB mínimo
- GPU recomendada para entrenamiento del modelo

## ⚙️ Instalación

1. **Clonar el repositorio**
```bash
git clone <url-repositorio>
cd <nombre-directorio>
```

2. **Crear y activar entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
Crear archivo `.env` con:
```
YOUTUBE_API_KEY=tu_api_key
MONGO_URI=mongodb://localhost:27017/
DATABASE_NAME=NLP
COLLECTION_NAME=lebels
```

## 📁 Estructura del Proyecto
```
proyecto/
├── data/                  # Datos de entrenamiento
│   └── youtoxic_english_1000.csv
├── models/               # Modelos entrenados
│   ├── final_model.h5
│   └── tokenizer.pickle
├── src/
│   ├── youtube_api.py   # Cliente API YouTube
│   ├── app.py          # Aplicación principal
│   └── rnn_antioverfitting.py  # Entrenamiento
├── requirements.txt
└── README.md
├── .dockerignore
└──docker-compose.yml 
├── .dockerfile
└──docker-compose.yml 
```

## 🚀 Uso

### Iniciar la Aplicación
```bash
streamlit run src/app.py
```

### Funcionalidades Principales
1. **Análisis Individual**
   - Analiza comentarios individuales
   - Muestra probabilidad de contenido ofensivo

2. **Análisis de Videos**
   - Analiza todos los comentarios de un video
   - Genera estadísticas y visualizaciones

3. **Monitoreo Continuo**
   - Seguimiento en tiempo real
   - Exportación de resultados

## 📦 Dependencias Principales
```
numpy>=1.26.2
pandas>=2.1.4
tensorflow>=2.15.0
streamlit>=1.29.0
pymongo>=4.6.1
nltk>=3.8.1
scikit-learn>=1.3.2
```

## 🔧 Configuración del Modelo
```python
MAX_WORDS = 15000
MAX_LEN = 128
EMBEDDING_DIM = 200
```

## 📊 Entrenamiento del Modelo
Para entrenar el modelo:
```bash
python src/rnn_antioverfitting.py
```

## 🤝 Contribución
1. Fork del proyecto
2. Crear rama de feature
   ```bash
   git checkout -b feature/NuevaFeature
   ```
3. Commit y push
4. Crear Pull Request
