import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager  # Para gestionar el driver automáticamente

# URL del video de YouTube
youtube_video_url = "https://www.youtube.com/watch?v=kuhhT_cBtFU&t=2s"

# Lista para almacenar los comentarios
data = []

# Configuración del driver usando Service
service = Service(ChromeDriverManager().install())  # Usamos webdriver_manager para manejar el chromedriver automáticamente

with Chrome(service=service) as driver:
    # Espera explícita
    wait = WebDriverWait(driver, 15)
    driver.get(youtube_video_url)

    # Esperar a que la página cargue completamente
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # Hacer scroll hacia abajo para cargar más comentarios
    for _ in range(5):  # Puedes ajustar el número de veces que hará scroll
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(3)  # Esperar a que se carguen los comentarios

    # Obtener todos los comentarios
    comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer #content")))

    for comment in comments:
        try:
            # Capturar el texto de cada comentario
            comment_text = comment.find_element(By.CSS_SELECTOR, "#content-text").text
            data.append(comment_text)
        except Exception as e:
            print(f"Error al obtener comentario: {e}")

    # Guardar los comentarios en un archivo de texto
    with open("comentarios_youtube.txt", "w", encoding="utf-8") as file:
        for comment in data:
            file.write(comment + "\n")

    print(f"Se han guardado {len(data)} comentarios en 'comentarios_youtube.txt'.")

