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


def preprocess(img: np.ndarray, use_clahe: bool = False) -> np.ndarray:
    """
    MEJORA: Pipeline configurable. Por defecto conserva la filosofía del
    paper (bordes de 1px) pero usa Canny que es más robusto que Sobel+threshold.
    
    - Sobel original tenía threshold fijo 30 que fallaba con variaciones de brillo.
    - Canny adapta sus umbrales y detecta bordes débiles sin romper continuos.
    - CLAHE disponible como opción para imágenes con contraste pobre (NO por defecto,
      porque puede introducir artefactos en imágenes ya bien contrastadas).
    - Sin dilatación: mantiene bordes de 1px como el paper. La Distance Transform
      en fitness.py compensa desplazamientos de sub-píxel.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Umbrales bajos para capturar bordes débiles sin fragmentar los continuos
    edge = cv2.Canny(blur, threshold1=30, threshold2=90)

    return edge


def get_edge_points(edge_map: np.ndarray) -> np.ndarray:
    """Devuelve array (N, 2) con coordenadas [x, y] de los píxeles de borde."""
    rows, cols = np.where(edge_map > 0)
    return np.column_stack((cols, rows)).astype(np.float64)


def annotate_image(img: np.ndarray, circles: list[dict]) -> np.ndarray:
    out = img.copy()
    for idx, c in enumerate(circles):
        x, y, r = int(c["x"]), int(c["y"]), int(c["r"])
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[idx % len(colors)]
        cv2.circle(out, (x, y), r, color, 2)
        cv2.circle(out, (x, y), 3, (255, 255, 255), -1)
        cv2.putText(out, f"#{idx+1} r={r}", (x - 30, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def ndarray_to_b64(img: np.ndarray) -> str:
    import base64
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()
