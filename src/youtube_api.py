import googleapiclient.discovery
from googleapiclient.errors import HttpError
import socket
import time
import logging
from typing import List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_comments(self, video_id: str, max_results: int = 100) -> List[str]:
        """Obtener comentarios de un video de YouTube.
        
        Args:
            video_id: ID del video de YouTube
            max_results: Número máximo de comentarios a obtener
            
        Returns:
            Lista de comentarios
        """
        for attempt in range(3):  # 3 intentos máximos
            try:
                # Aumentar el timeout
                socket.setdefaulttimeout(120)  # 2 minutos
                
                # Crear el cliente de YouTube
                youtube = googleapiclient.discovery.build(
                    "youtube",
                    "v3",
                    developerKey=self.api_key,
                    cache_discovery=False
                )
                
                # Solicitar los comentarios
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=max_results,
                    textFormat="plainText"
                )
                
                response = request.execute()
                
                if 'items' not in response or not response['items']:
                    raise ValueError("No se encontraron comentarios para este video")
                
                comments = []
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    if comment.strip():
                        comments.append(comment)
                
                if not comments:
                    raise ValueError("No se encontraron comentarios válidos")
                    
                return comments
                
            except socket.timeout:
                if attempt < 2:  # Si no es el último intento
                    time.sleep(2 ** attempt)  # Espera exponencial
                    continue
                raise ValueError(
                    "La conexión con YouTube ha excedido el tiempo de espera. "
                    "Por favor, verifica tu conexión a internet."
                )
                
            except HttpError as e:
                if e.resp.status in [403, 429]:
                    raise ValueError("Se ha excedido el límite de la API")
                elif e.resp.status == 404:
                    raise ValueError("Video no encontrado o no disponible")
                else:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Error de la API de YouTube: {str(e)}")
                    
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise ValueError(f"Error inesperado: {str(e)}")
        
        raise ValueError("No se pudieron obtener los comentarios después de varios intentos")