"""
Aquí están todas las funciones que preparan la imagen antes de pasársela
al algoritmo genético, y las que dibujan y exportan el resultado final.
"""

import math
import cv2
import numpy as np
import base64

# ── Configuración general ─────────────────────────────────────────────────────
KERNEL_DESENFOQUE    = (5, 5)
MAX_PUNTOS_BORDE     = 5000    # Aumentado para no perder círculos pequeños

# ── Configuración de la supresión de líneas rectas ────────────────────────────
# MEJORA PROFUNDA: estos valores ahora son MÁS CONSERVADORES.
# El problema original: la supresión de líneas rectas era tan agresiva que
# eliminaba bordes de círculos grandes (cuyo borde localmente parece recto)
# y bordes de círculos cercanos a líneas.
BRECHA_MAXIMA_LINEA  = 4     # Antes 8, luego 5. Ahora 4: menos conexiones falsas
GROSOR_MASCARA_LINEA = 3     # Antes 7, luego 4. Ahora 3: borra menos ancho

# ── Cargar imagen ─────────────────────────────────────────────────────────────

def cargar_imagen_desde_bytes(datos_imagen: bytes) -> np.ndarray:
    arreglo_bytes = np.frombuffer(datos_imagen, np.uint8)
    imagen = cv2.imdecode(arreglo_bytes, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError("No se pudo decodificar la imagen. Verifica que el formato sea válido.")
    return imagen


# ── Eliminar líneas rectas del mapa de bordes ─────────────────────────────────

def suprimir_lineas_rectas(mapa_bordes: np.ndarray, forma_imagen: tuple,
                           habilitar: bool = True) -> np.ndarray:
    """
    MEJORA PROFUNDA: la supresión de líneas ahora es MÁS CONSERVADORA.

    El problema original: con GROSOR_MASCARA_LINEA=7 y maxLineGap=8, esta función
    eliminaba bordes de círculos legítimos porque:
    1. Los bordes de círculos grandes tienen segmentos LOCALMENTE rectos
    2. HoughLinesP conectaba puntos separados por 8px, formando "líneas" falsas
    3. La máscara de 7px de grosro destruía bordes de círculos cercanos

    Ahora:
    - maxLineGap=4: no conecta segmentos separados por mucho espacio
    - GROSOR_MASCARA_LINEA=3: borra solo lo necesario
    - longitud_min más conservadora: no detecta bordes curvos como rectos
    """
    if not habilitar:
        return mapa_bordes

    alto, ancho = forma_imagen[:2]
    diagonal = math.sqrt(alto ** 2 + ancho ** 2)

    # Mucho más conservador: solo líneas MUY largas y MUY rectas
    longitud_min = max(80, int(diagonal * 0.20))
    umbral_votos = max(30, int(longitud_min * 0.6))

    lineas = cv2.HoughLinesP(
        mapa_bordes,
        rho=1,
        theta=np.pi / 180,
        threshold=umbral_votos,
        minLineLength=longitud_min,
        maxLineGap=BRECHA_MAXIMA_LINEA,
    )

    if lineas is None:
        return mapa_bordes

    mascara = np.zeros_like(mapa_bordes)
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(mascara, (x1, y1), (x2, y2), 255, GROSOR_MASCARA_LINEA)

    return cv2.bitwise_and(mapa_bordes, cv2.bitwise_not(mascara))


# ── Preparar la imagen para el GA ────────────────────────────────────────────

def preprocesar(imagen: np.ndarray, suprimir_lineas: bool = True) -> np.ndarray:
    """
    MEJORA PROFUNDA: Preprocesado adaptativo con menos destrucción de bordes.

    El problema original: Canny con umbrales basados en percentiles funcionaba
    bien para imágenes "típicas", pero en imágenes con:
    - Bajo contraste: los percentiles bajos generaban demasiados bordes espurios
    - Alto ruido: la mediana era muy alta, eliminando bordes reales
    - Círculos de color similar al fondo: los gradientes eran débiles

    Solución: usar un enfoque más robusto basado en la magnitud del gradiente.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, KERNEL_DESENFOQUE, 0)

    # Gradientes
    gx = cv2.Sobel(suavizada, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(suavizada, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.hypot(gx, gy)

    # Umbrales adaptativos más robustos
    valores_grad = magnitud[magnitud > 0].ravel()
    if len(valores_grad) > 100:
        # Usar percentiles más conservadores
        p25 = float(np.percentile(valores_grad, 25))
        p75 = float(np.percentile(valores_grad, 75))
        umbral_bajo = max(15, int(p25 * 0.5))
        umbral_alto = max(umbral_bajo + 20, int(p75 * 0.8))
    else:
        umbral_bajo, umbral_alto = 30, 100

    mapa_bordes = cv2.Canny(suavizada, umbral_bajo, umbral_alto)
    mapa_bordes = suprimir_lineas_rectas(mapa_bordes, imagen.shape, suprimir_lineas)

    return mapa_bordes


# ── Extraer los puntos de borde ───────────────────────────────────────────────

def obtener_puntos_borde(mapa_bordes: np.ndarray) -> np.ndarray:
    """
    MEJORA PROFUNDA: estratificación espacial para preservar círculos pequeños.

    El problema original: muestreo aleatorio de 5000 puntos podía eliminar
    TODOS los puntos de un círculo pequeño si había muchos bordes en otras
    partes de la imagen.

    Solución: dividir la imagen en una cuadrícula y asegurar que cada celda
    contribuya proporcionalmente. Esto preserva círculos pequeños incluso
    cuando hay muchos bordes en otras zonas.
    """
    filas, columnas = np.where(mapa_bordes > 0)
    puntos_borde = np.column_stack((columnas, filas)).astype(np.float64)

    n = len(puntos_borde)
    if n <= MAX_PUNTOS_BORDE:
        return puntos_borde

    # Estratificación espacial: dividir en cuadrícula 4x4
    alto, ancho = mapa_bordes.shape
    celdas = []
    for i in range(4):
        for j in range(4):
            y0, y1 = int(alto * i / 4), int(alto * (i + 1) / 4)
            x0, x1 = int(ancho * j / 4), int(ancho * (j + 1) / 4)
            mask = ((puntos_borde[:, 0] >= x0) & (puntos_borde[:, 0] < x1) &
                    (puntos_borde[:, 1] >= y0) & (puntos_borde[:, 1] < y1))
            celdas.append(puntos_borde[mask])

    # Tomar muestras proporcionales de cada celda
    puntos_muestra = []
    puntos_por_celda = MAX_PUNTOS_BORDE // 16

    for celda in celdas:
        if len(celda) == 0:
            continue
        if len(celda) <= puntos_por_celda:
            puntos_muestra.append(celda)
        else:
            indices = np.random.choice(len(celda), puntos_por_celda, replace=False)
            puntos_muestra.append(celda[indices])

    # Si sobran puntos, llenar con aleatorios globales
    resultado = np.vstack(puntos_muestra) if puntos_muestra else np.array([])
    if len(resultado) < MAX_PUNTOS_BORDE and n > 0:
        faltan = MAX_PUNTOS_BORDE - len(resultado)
        indices_extra = np.random.choice(n, min(faltan, n), replace=False)
        resultado = np.vstack([resultado, puntos_borde[indices_extra]])

    return resultado


# ── Dibujar los círculos detectados sobre la imagen ──────────────────────────

def anotar_imagen(imagen: np.ndarray, circulos: list) -> np.ndarray:
    imagen_anotada = imagen.copy()

    for circulo in circulos:
        centro_x = int(circulo["x"])
        centro_y = int(circulo["y"])
        radio = int(circulo["r"])

        cv2.circle(imagen_anotada, (centro_x, centro_y), radio, (255, 255, 255), 2)
        cv2.circle(imagen_anotada, (centro_x, centro_y), 3, (200, 200, 200), -1)

    return imagen_anotada


# ── Convertir la imagen a texto para enviarla al frontend ─────────────────────

def imagen_a_base64(imagen: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", imagen)
    cadena_b64 = base64.b64encode(buffer.tobytes()).decode()
    return cadena_b64
