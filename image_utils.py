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
    """Convierte a escala de grises y aplica Sobel para obtener mapa de bordes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = np.uint8(255 * mag / mag.max())
    _, edge = cv2.threshold(mag, 30, 255, cv2.THRESH_BINARY)
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
