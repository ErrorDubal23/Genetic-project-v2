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
    MEJORA: Pipeline de preprocesado más robusto.
    - Conversión a escala de grises
    - CLAHE para normalizar contraste (maneja variaciones de brillo)
    - GaussianBlur para reducir ruido
    - Canny en lugar de Sobel + threshold manual (mejor para bordes débiles)
    - Dilatación ligera para conectar bordes fragmentados (ayuda con oclusión)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # MEJORA 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Normaliza contraste localmente sin amplificar ruido excesivamente
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Suavizado gaussiano para reducir ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # MEJORA 2: Canny edge detector en lugar de Sobel + threshold
    # Canny es más robusto: usa gradiente + supresión no-máxima + histéresis
    edge = cv2.Canny(blur, threshold1=50, threshold2=150)

    # MEJORA 3: Dilatación ligera para conectar bordes fragmentados
    # Ayuda cuando hay oclusión parcial o bordes poco definidos
    kernel = np.ones((2, 2), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)

    return edge


def get_edge_points(edge_map: np.ndarray) -> np.ndarray:
    """Devuelve array (N, 2) con coordenadas [col, row] de los píxeles de borde."""
    rows, cols = np.where(edge_map > 0)
    return np.column_stack((cols, rows)).astype(np.float64)


def annotate_image(img: np.ndarray, circles: list[dict]) -> np.ndarray:
    out = img.copy()
    for idx, c in enumerate(circles):
        x, y, r = int(c["x"]), int(c["y"]), int(c["r"])
        # Colores distintos para cada círculo detectado
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][idx % 5]
        cv2.circle(out, (x, y), r, color, 2)
        cv2.circle(out, (x, y), 3, (255, 255, 255), -1)
        cv2.putText(out, f"#{idx+1} r={r}", (x - 30, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def ndarray_to_b64(img: np.ndarray) -> str:
    import base64
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()
