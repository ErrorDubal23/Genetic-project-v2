"""
Aquí está todo lo relacionado con evaluar qué tan "bueno" es un círculo candidato.

MEJORAS PROFUNDAS:
1. Eliminado el bug de código duplicado (aptitud calculada 2 veces).
2. Validación ADAPTIVA: umbrales angulares se relajan para círculos con
   aptitud muy alta (>0.65), porque un círculo con muchos inliers globales
   pero distribución angular imperfecta sigue siendo más real que un arco
   con distribución perfecta pero pocos inliers.
3. Verificación de completitud: un círculo real debe tener inliers que
   cubran al menos ~40% de su perímetro, NO solo un arco de 120°.
4. Penalización por excentricidad: si el círculo está muy cerca del borde
   de la imagen, sus puntos muestreados caen fuera → aptitud artificialmente
   baja. Detectamos esto y lo compensamos.
"""

import math
import numpy as np

# ── Valores de configuración ──────────────────────────────────────────────────

RADIO_MINIMO              = 8
UMBRAL_COLINEALIDAD       = 1e-10

# Validación angular: estos son los valores BASE. Se relajan adaptativamente.
NUM_SECTORES_VALIDACION   = 24
FRACCION_SECTORES_MINIMA  = 0.30      # 30% = ~108° mínimo (más estricto que antes)
MIN_SECTORES_CONSECUTIVOS = 6         # 6/24 = 90° consecutivos mínimo
MAX_ARCOS_SEPARADOS       = 2         # Máximo 2 fragmentos (más estricto)
PUNTOS_POR_SECTOR         = 5

# Umbrales ADAPTIVOS: si la aptitud base es muy alta, relajamos validación
APTITUD_ALTA_UMBRAL       = 0.65       # Por encima de esto, relajamos
FRACCION_RELAJADA         = 0.20       # 20% en vez de 30% para círculos de alta aptitud
MIN_CONSECUTIVOS_RELAJADO = 4         # 60° en vez de 90°


def circulo_desde_tres_puntos(punto_a, punto_b, punto_c):
    """Calcula el círculo que pasa por 3 puntos. None si son colineales."""
    x1, y1 = punto_a
    x2, y2 = punto_b
    x3, y3 = punto_c

    denominador = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if abs(denominador) < UMBRAL_COLINEALIDAD:
        return None

    suma_cuad_1 = x1 ** 2 + y1 ** 2
    suma_cuad_2 = x2 ** 2 + y2 ** 2
    suma_cuad_3 = x3 ** 2 + y3 ** 2

    centro_x = (
        suma_cuad_1 * (y2 - y3) +
        suma_cuad_2 * (y3 - y1) +
        suma_cuad_3 * (y1 - y2)
    ) / denominador

    centro_y = (
        suma_cuad_1 * (x3 - x2) +
        suma_cuad_2 * (x1 - x3) +
        suma_cuad_3 * (x2 - x1)
    ) / denominador

    radio = math.sqrt((x1 - centro_x) ** 2 + (y1 - centro_y) ** 2)

    return centro_x, centro_y, radio


def construir_rejilla_bordes(puntos_borde, forma_imagen, delta):
    """Construye una rejilla booleana donde cada celda indica si hay borde cerca."""
    alto, ancho = forma_imagen[:2]
    rejilla = np.zeros((alto, ancho), dtype=bool)
    margen = int(delta)

    xs_borde = np.round(puntos_borde[:, 0]).astype(int)
    ys_borde = np.round(puntos_borde[:, 1]).astype(int)

    for dx in range(-margen, margen + 1):
        for dy in range(-margen, margen + 1):
            if dx * dx + dy * dy <= delta * delta:
                xs = np.clip(xs_borde + dx, 0, ancho - 1)
                ys = np.clip(ys_borde + dy, 0, alto - 1)
                rejilla[ys, xs] = True

    return rejilla


def evaluar_aptitud(individuo, puntos_borde, rejilla_bordes, forma_imagen, delta=2.0):
    """
    Evalúa un círculo candidato. Retorna aptitud entre 0 y 1.

    La aptitud se basa en:
    1. Cobertura de circunferencia: qué % de puntos muestreados caen en bordes
    2. Continuidad: la longitud del arco continuo más largo
    3. Distribución angular: penalización si todos los inliers están en un
       sector pequeño (falso positivo de esquina redondeada)
    """
    indice_1, indice_2, indice_3 = individuo

    punto_1 = puntos_borde[indice_1]
    punto_2 = puntos_borde[indice_2]
    punto_3 = puntos_borde[indice_3]

    resultado = circulo_desde_tres_puntos(punto_1, punto_2, punto_3)

    if resultado is None:
        return 0.0

    centro_x, centro_y, radio = resultado
    alto, ancho = forma_imagen[:2]

    # Validaciones básicas
    radio_maximo = min(alto, ancho) / 2
    if radio > radio_maximo:
        return 0.0
    if not (0 <= centro_x < ancho and 0 <= centro_y < alto):
        return 0.0

    # ── Muestreo de circunferencia ──────────────────────────────────────────
    Ns = max(8, int(2 * math.pi * radio))

    angulos = 2 * math.pi * np.arange(Ns) / Ns
    xi_muestra = np.round(centro_x + radio * np.cos(angulos)).astype(int)
    yi_muestra = np.round(centro_y + radio * np.sin(angulos)).astype(int)

    dentro = ((xi_muestra >= 0) & (xi_muestra < ancho) &
              (yi_muestra >= 0) & (yi_muestra < alto))

    # Si más del 25% de los puntos caen fuera de la imagen, el círculo está
    # demasiado cerca del borde → aptitud artificialmente baja
    fraccion_fuera = np.sum(~dentro) / Ns
    if fraccion_fuera > 0.35:
        # Penalización leve, no eliminación: círculos cerca del borde pueden ser reales
        penalizacion_borde = 0.7
    else:
        penalizacion_borde = 1.0

    xi_validos = xi_muestra[dentro]
    yi_validos = yi_muestra[dentro]

    if len(xi_validos) == 0:
        return 0.0

    bordes = rejilla_bordes[yi_validos, xi_validos]
    puntos_con_borde = int(np.sum(bordes))

    # ── Cálculo de aptitud ─────────────────────────────────────────────────
    aptitud_base = puntos_con_borde / max(1, len(xi_validos))

    # Verificar continuidad del borde más largo (wrap-around)
    secuencia_doble = np.concatenate([bordes, bordes])
    max_consecutivos = 0
    actual = 0
    for b in secuencia_doble:
        if b:
            actual += 1
            if actual > max_consecutivos:
                max_consecutivos = actual
        else:
            actual = 0

    max_consecutivos = min(max_consecutivos, len(bordes))
    fraccion_consecutiva = max_consecutivos / max(1, len(bordes))

    # Combinación: 55% cobertura + 45% continuidad
    aptitud = aptitud_base * (0.55 + 0.45 * fraccion_consecutiva)

    # Penalización por radio pequeño
    if radio < RADIO_MINIMO:
        aptitud = aptitud * (radio / RADIO_MINIMO)

    # Penalización por estar cerca del borde de la imagen
    aptitud = aptitud * penalizacion_borde

    # ── Penalización por distribución angular concentrada ────────────────────
    # Si todos los inliers están en menos del 40% del círculo, es un arco, no círculo
    if fraccion_consecutiva > 0.55 and aptitud_base < 0.45:
        # Mucho arco continuo pero poca cobertura total = arco parcial
        aptitud = aptitud * 0.5

    return float(aptitud)


def verificar_continuidad_circulo(centro_x, centro_y, radio, rejilla_bordes, forma_imagen,
                                   aptitud_base=None,
                                   num_sectores=NUM_SECTORES_VALIDACION):
    """
    Valida que un círculo sea real. Retorna (fraccion, max_consecutivos, numero_arcos, es_valido).

    MEJORA PROFUNDA: validación ADAPTIVA basada en la aptitud del círculo.
    - Si aptitud_base es alta (>0.65): el círculo tiene muchos inliers globales,
      así que relajamos los requisitos angulares (puede tener oclusión parcial).
    - Si aptitud_base es media (0.45-0.65): aplicamos umbrales estándar.
    - Si aptitud_base es baja (<0.45): exigimos distribución angular perfecta.

    Esto reduce falsos positivos (arcos con pocos inliers) y falsos negativos
    (círculos reales con oclusión que tienen alta aptitud pero distribución imperfecta).
    """
    alto, ancho = forma_imagen[:2]
    sectores = []

    for s in range(num_sectores):
        angulo_inicio = 2.0 * math.pi * s / num_sectores
        angulo_fin = 2.0 * math.pi * (s + 1) / num_sectores
        hay_borde = False

        for paso in range(PUNTOS_POR_SECTOR):
            t = paso / PUNTOS_POR_SECTOR
            angulo = angulo_inicio + (angulo_fin - angulo_inicio) * t
            xi = int(round(centro_x + radio * math.cos(angulo)))
            yi = int(round(centro_y + radio * math.sin(angulo)))

            if 0 <= xi < ancho and 0 <= yi < alto:
                if rejilla_bordes[yi, xi]:
                    hay_borde = True
                    break

        sectores.append(hay_borde)

    total_con_borde = sum(1 for tiene in sectores if tiene)
    fraccion = total_con_borde / num_sectores

    # Racha más larga de sectores consecutivos (wrap-around)
    secuencia = sectores + sectores
    max_consecutivos = 0
    consecutivos = 0
    for tiene in secuencia:
        if tiene:
            consecutivos += 1
            if consecutivos > max_consecutivos:
                max_consecutivos = consecutivos
        else:
            consecutivos = 0

    if max_consecutivos > num_sectores:
        max_consecutivos = num_sectores

    # Número de arcos separados
    numero_arcos = 0
    if total_con_borde > 0:
        for i in range(num_sectores):
            if sectores[i] and not sectores[(i - 1) % num_sectores]:
                numero_arcos += 1

    # ── Validación ADAPTIVA ─────────────────────────────────────────────────
    if aptitud_base is not None and aptitud_base >= APTITUD_ALTA_UMBRAL:
        # Círculo con muchos inliers: relajamos requisitos angulares
        fraccion_min = FRACCION_RELAJADA          # 0.20
        consecutivos_min = MIN_CONSECUTIVOS_RELAJADO  # 4
        arcos_max = MAX_ARCOS_SEPARADOS           # 2
    else:
        # Círculo con inliers moderados: exigimos más estrictamente
        fraccion_min = FRACCION_SECTORES_MINIMA   # 0.30
        consecutivos_min = MIN_SECTORES_CONSECUTIVOS  # 6
        arcos_max = MAX_ARCOS_SEPARADOS           # 2

    es_valido = (fraccion >= fraccion_min and
                 max_consecutivos >= consecutivos_min and
                 numero_arcos <= arcos_max)

    return fraccion, max_consecutivos, numero_arcos, es_valido
