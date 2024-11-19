# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements y del proyecto
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código al contenedor
COPY src/ ./src
COPY models/ ./models

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV TF_ENABLE_ONEDNN_OPTS=0

# Exponer el puerto para Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "src/app.py"]
