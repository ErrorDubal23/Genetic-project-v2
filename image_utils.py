import cv2
import numpy as np
from PIL import Image
import io


def load_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    MEJORA: Canny en lugar de Sobel + threshold fijo.
    
    El código original usaba Sobel con threshold binario fijo (30).
    Problema: imágenes con bajo brillo/contraste no generaban gradientes
    superiores a 30 y perdían todos sus bordes → falsos negativos totales.
    
    Canny usa histéresis adaptativa (dos umbrales) + supresión no-máxima,
    detectando bordes débiles conectados a fuertes sin romper contornos.
    Internamente usa Sobel, así que la filosofía del paper se conserva.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Umbrales bajos para capturar bordes débiles sin fragmentar continuos
    edge = cv2.Canny(blur, threshold1=30, threshold2=90)
    return edge


def get_edge_points(edge_map: np.ndarray) -> np.ndarray:
    """Devuelve array (N, 2) con coordenadas [col, row] de los píxeles de borde."""
    rows, cols = np.where(edge_map > 0)
    return np.column_stack((cols, rows)).astype(np.float64)


def annotate_image(img: np.ndarray, circles: list[dict]) -> np.ndarray:
    out = img.copy()
    for c in circles:
        x, y, r = int(c["x"]), int(c["y"]), int(c["r"])
        cv2.circle(out, (x, y), r, (255, 255, 255), 2)
        cv2.circle(out, (x, y), 3, (200, 200, 200), -1)
    return out


def ndarray_to_b64(img: np.ndarray) -> str:
    import base64
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()
