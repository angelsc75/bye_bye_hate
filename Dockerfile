# Usar Python 3.10 como base
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema y Chrome
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    xvfb \
    chromium \
    chromium-driver \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    DISPLAY=:99 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Copiar requirements.txt primero
COPY requirements.txt ./

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar recursos NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Crear directorios necesarios
RUN mkdir -p models src

# Copiar código
COPY src/ ./src/
COPY start.sh ./

# Hacer ejecutable el script de inicio
RUN chmod +x start.sh

# Exponer puerto para Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["./start.sh"]