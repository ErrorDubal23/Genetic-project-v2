"""
Aquí están todas las funciones que preparan la imagen antes de pasársela
al algoritmo genético, y las que dibujan y exportan el resultado final.
"""

import math
import cv2
import numpy as np
import base64

# ── Configuración general ─────────────────────────────────────────────────────
KERNEL_DESENFOQUE    = (5, 5)  # Tamaño del filtro de suavizado para reducir ruido
MAX_PUNTOS_BORDE     = 5000    # Límite de puntos que le pasamos al GA.
                                # MEJORA: antes 3000, ahora 5000.
                                # Con solo 3000, círculos pequeños en imágenes
                                # densas se perdían porque sus puntos eran
                                # eliminados en el muestreo aleatorio.

# ── Configuración de la supresión de líneas rectas ────────────────────────────
BRECHA_MAXIMA_LINEA  = 5   # MEJORA: antes 8, ahora 5.
                            # Con 8px, HoughLinesP conectaba segmentos
                            # separados por mucho espacio, tratando bordes
                            # curvos fragmentados como una "línea recta"
                            # continua y eliminándolos por error.

GROSOR_MASCARA_LINEA = 4   # MEJORA: antes 7, ahora 4.
                            # 7px borraba demasiado ancho, destruyendo
                            # bordes de círculos grandes que localmente
                            # tienen segmentos casi rectos. 4px es suficiente
                            # para tapar líneas reales sin dañar curvas.


# ── Cargar imagen ─────────────────────────────────────────────────────────────

def cargar_imagen_desde_bytes(datos_imagen: bytes) -> np.ndarray:
    """
    Convierte los bytes del archivo subido en una imagen que podamos procesar.
    Si el archivo no es una imagen válida, lanza un error explicando el problema.
    """
    arreglo_bytes = np.frombuffer(datos_imagen, np.uint8)
    imagen = cv2.imdecode(arreglo_bytes, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError("No se pudo decodificar la imagen. Verifica que el formato sea válido.")

    return imagen


# ── Eliminar líneas rectas del mapa de bordes ─────────────────────────────────

def suprimir_lineas_rectas(mapa_bordes: np.ndarray, forma_imagen: tuple) -> np.ndarray:
    """
    Detecta y borra los segmentos de línea recta del mapa de bordes, para que
    el GA solo vea bordes curvos (que son los que forman círculos).

    Los lados de triángulos, rectángulos y pentágonos son líneas rectas.
    Los bordes de los círculos son curvos. Esta función elimina las líneas rectas
    antes de que el GA las confunda con partes de un círculo.

    El umbral de longitud mínima se calcula automáticamente según el tamaño de
    la imagen, porque en imágenes grandes los círculos también son grandes y
    sus bordes parecen más rectos localmente. Si usáramos un umbral fijo,
    eliminaríamos bordes de círculos grandes por error.
    """
    alto, ancho = forma_imagen[:2]
    diagonal    = math.sqrt(alto ** 2 + ancho ** 2)

    # MEJORA: antes diagonal*0.18 = 144px para 640x480. Con esa longitud,
    # círculos grandes (r > 80px) podían tener bordes locales detectados
    # como líneas rectas. Ahora 0.15 = 120px, más conservador.
    longitud_min =  max(50, int(diagonal * 0.15))
    umbral_votos = max(20, int(longitud_min * 0.55))

    lineas = cv2.HoughLinesP(
        mapa_bordes,
        rho=1,
        theta=np.pi / 180,
        threshold=umbral_votos,
        minLineLength=longitud_min,
        maxLineGap=BRECHA_MAXIMA_LINEA,
    )

    if lineas is None:
        return mapa_bordes  # No se encontraron líneas rectas, nada que borrar

    # Dibujamos un "borrador" sobre cada línea detectada y lo aplicamos
    mascara = np.zeros_like(mapa_bordes)
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(mascara, (x1, y1), (x2, y2), 255, GROSOR_MASCARA_LINEA)

    return cv2.bitwise_and(mapa_bordes, cv2.bitwise_not(mascara))


# ── Preparar la imagen para el GA ────────────────────────────────────────────

def preprocesar(imagen: np.ndarray) -> np.ndarray:
    """
    Transforma la imagen original en un mapa de bordes que el GA pueda usar.

    Los pasos son:
      1. Convertir a escala de grises (el GA no necesita color).
      2. Suavizar para reducir el ruido de la textura.
      3. Detectar los bordes con Canny.
      4. Borrar las líneas rectas para que queden principalmente bordes curvos.

    Los umbrales de Canny los calculamos a partir de la fuerza real de los bordes
    en la imagen (no del brillo de los píxeles). Esto hace que funcione bien tanto
    en imágenes oscuras como en imágenes con fondo blanco, donde los círculos de
    colores claros tienen bordes más suaves que los oscuros.
    """
    gris      = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # MEJORA: sigma=0 en lugar de sigma=2.
    # Con sigma=2, GaussianBlur difuminaba demasiado los bordes finos,
    # especialmente de círculos pequeños o con poco contraste.
    # sigma=0 calcula automáticamente según el tamaño del kernel (5x5),
    # preservando bordes más finos sin aumentar el ruido.
    suavizada = cv2.GaussianBlur(gris, KERNEL_DESENFOQUE, 0)

    # Calculamos qué tan fuertes son los gradientes (cambios de color) en la imagen
    gx = cv2.Sobel(suavizada, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(suavizada, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.hypot(gx, gy)

    # Usamos los percentiles de esa fuerza para fijar los umbrales de Canny
    valores_grad = magnitud[magnitud > 0].ravel()
    if len(valores_grad) > 100:
        mediana_grad = float(np.median(valores_grad))
        p85_grad     = float(np.percentile(valores_grad, 85))
        umbral_bajo  = max(10, int(mediana_grad * 0.4))
        umbral_alto  = max(umbral_bajo + 30, int(p85_grad))
    else:
        umbral_bajo, umbral_alto = 30, 100

    mapa_bordes = cv2.Canny(suavizada, umbral_bajo, umbral_alto)
    mapa_bordes = suprimir_lineas_rectas(mapa_bordes, imagen.shape)

    return mapa_bordes


# ── Extraer los puntos de borde ───────────────────────────────────────────────

def obtener_puntos_borde(mapa_bordes: np.ndarray) -> np.ndarray:
    """
    Convierte el mapa de bordes en una lista de coordenadas (x, y).
    Si hay demasiados puntos, tomamos una muestra aleatoria para que el GA
    no se vuelva lento.
    """
    filas, columnas = np.where(mapa_bordes > 0)
    puntos_borde    = np.column_stack((columnas, filas)).astype(np.float64)

    if len(puntos_borde) > MAX_PUNTOS_BORDE:
        indices_muestra = np.random.choice(len(puntos_borde), MAX_PUNTOS_BORDE, replace=False)
        puntos_borde    = puntos_borde[indices_muestra]

    return puntos_borde


# ── Dibujar los círculos detectados sobre la imagen ──────────────────────────

def anotar_imagen(imagen: np.ndarray, circulos: list) -> np.ndarray:
    """
    Dibuja cada círculo detectado encima de la imagen original para visualizarlo.
    """
    imagen_anotada = imagen.copy()

    for circulo in circulos:
        centro_x = int(circulo["x"])
        centro_y = int(circulo["y"])
        radio    = int(circulo["r"])

        cv2.circle(imagen_anotada, (centro_x, centro_y), radio, (255, 255, 255), 2)
        cv2.circle(imagen_anotada, (centro_x, centro_y), 3,     (200, 200, 200), -1)

    return imagen_anotada


# ── Convertir la imagen a texto para enviarla al frontend ─────────────────────

def imagen_a_base64(imagen: np.ndarray) -> str:
    """
    Convierte la imagen a formato base64 para que el frontend pueda mostrarla
    directamente sin necesidad de guardar un archivo.
    """
    _, buffer  = cv2.imencode(".png", imagen)
    cadena_b64 = base64.b64encode(buffer.tobytes()).decode()
    return cadena_b64
