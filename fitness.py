"""
Aquí está todo lo relacionado con evaluar qué tan "bueno" es un círculo candidato.
El algoritmo genético usa estas funciones para saber si un individuo vale la pena
o no, y al final para confirmar que lo que encontró es un círculo real.

Basado en: Ayala-Ramírez et al. (2006), secciones 2.2 y 2.3.

MEJORAS CLAVE:
- El código original ejecutaba evaluar_aptitud DOS VECES para cada individuo
  (líneas duplicadas), calculando aptitud_base, aptitud, y luego
  redefiniendo bordes, puntos_con_borde, aptitud_base DE NUEVO.
  Esto no cambiaba el resultado pero DOBLABA el tiempo de evaluación
  de cada individuo, haciendo todo el GA 2× más lento sin beneficio.
- aptitud_base se calculaba dos veces, aptitud se sobreescribía sin usar
  el primer valor. Código muerto eliminado.
- La penalización por fragmentación (fraccion_consecutiva) ahora se aplica
  correctamente como multiplicador suave, no como reemplazo.
"""

import math
import numpy as np

# ── Valores de configuración ──────────────────────────────────────────────────

RADIO_MINIMO              = 8    # MEJORA: antes 10, ahora 8.
                                  # 10px eliminaba círculos pequeños pero reales
                                  # en imágenes de baja resolución. 8px es más
                                  # permisivo sin caer en ruido.

UMBRAL_COLINEALIDAD       = 1e-10

NUM_SECTORES_VALIDACION   = 24   # MEJORA: antes 18, ahora 24.
                                  # Con solo 18 sectores (20° cada uno), círculos
                                  # con oclusión parcial o bordes irregulares
                                  # podían no cumplir el mínimo de 6 sectores
                                  # consecutivos por artefactos de muestreo.
                                  # 24 sectores (15° cada uno) = resolución más fina
                                  # para detectar arcos continuos reales.

FRACCION_SECTORES_MINIMA  = 0.25 # MEJORA: antes 0.30, ahora 0.25.
                                  # 30% = 108° mínimo. Círculos con oclusión
                                  # parcial significativa o bordes degradados
                                  # podían tener solo 90-100° visibles y ser
                                  # descartados como falsos negativos.
                                  # 25% = 90° mínimo, más realista para círculos
                                  # reales parcialmente ocluidos.

MIN_SECTORES_CONSECUTIVOS = 5    # MEJORA: antes 6, ahora 5.
                                  # 6 sectores con 18 = 120°. Con 24 sectores,
                                  # 5 consecutivos = 75°. Un círculo real
                                  # aunque esté tapado por la mitad siempre tiene
                                  # al menos 90-180° continuos. 75° es seguro
                                  # para descartar polígonos pero no círculos.

MAX_ARCOS_SEPARADOS       = 3    # MEJORA: antes 2, ahora 3.
                                  # Un círculo parcialmente ocluido por 2 objetos
                                  # puede fragmentarse en 3 arcos separados
                                  # (ej: círculo tapado arriba y abajo).
                                  # Con límite 2, estos círculos válidos eran
                                  # descartados como falsos negativos.

PUNTOS_POR_SECTOR         = 5

# ── Calcular el círculo que pasa por 3 puntos ────────────────────────────────

def circulo_desde_tres_puntos(punto_a, punto_b, punto_c):
    """
    Dado cualquier trio de puntos, calcula el único círculo que pasa por los tres.
    Si los puntos están en línea recta, no existe tal círculo y retorna None.
    Retorna (centro_x, centro_y, radio).
    """
    x1, y1 = punto_a
    x2, y2 = punto_b
    x3, y3 = punto_c

    denominador = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if abs(denominador) < UMBRAL_COLINEALIDAD:
        return None  # Los puntos están en línea recta, no hay círculo posible

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


# ── Mapa de bordes con margen ─────────────────────────────────────────────────

def construir_rejilla_bordes(puntos_borde, forma_imagen, delta):
    """
    Construye una cuadrícula del tamaño de la imagen donde cada celda dice
    si hay un píxel de borde cerca (a menos de 'delta' píxeles de distancia).

    Esto nos permite preguntar "¿hay borde aquí?" en tiempo constante durante
    la evaluación, en vez de buscar en toda la lista de bordes cada vez.
    Se calcula una sola vez al inicio de cada búsqueda.
    """
    alto, ancho = forma_imagen[:2]
    rejilla     = np.zeros((alto, ancho), dtype=bool)
    margen      = int(delta)

    xs_borde = np.round(puntos_borde[:, 0]).astype(int)
    ys_borde = np.round(puntos_borde[:, 1]).astype(int)

    # Marcamos como "tiene borde cerca" todos los píxeles dentro del radio delta
    for dx in range(-margen, margen + 1):
        for dy in range(-margen, margen + 1):
            if dx * dx + dy * dy <= delta * delta:
                xs = np.clip(xs_borde + dx, 0, ancho - 1)
                ys = np.clip(ys_borde + dy, 0, alto  - 1)
                rejilla[ys, xs] = True

    return rejilla


# ── Función de aptitud: qué tan bien encaja el círculo ───────────────────────

def evaluar_aptitud(individuo, puntos_borde, rejilla_bordes, forma_imagen,
                    delta=2.0):
    """
    Le pone una nota de 0 a 1 al círculo candidato.

    La idea es simple: tomamos puntos uniformemente repartidos alrededor de la
    circunferencia del círculo y contamos cuántos de ellos caen sobre un borde
    real en la imagen. Si la mayoría cae en bordes, la nota es alta. Si casi
    ninguno cae, es baja.

    Un valor de 1.0 significa que toda la circunferencia tiene borde.
    Un valor de 0.45 significa que el 45% de la circunferencia tiene borde.
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

    # Descartamos círculos absurdamente grandes o con centro fuera de la imagen
    radio_maximo = min(alto, ancho) / 2
    if radio > radio_maximo:
        return 0.0
    if not (0 <= centro_x < ancho and 0 <= centro_y < alto):
        return 0.0

    # Repartimos puntos alrededor de la circunferencia (uno por píxel del perímetro)
    Ns = max(8, int(2 * math.pi * radio))

    angulos    = 2 * math.pi * np.arange(Ns) / Ns
    xi_muestra = np.round(centro_x + radio * np.cos(angulos)).astype(int)
    yi_muestra = np.round(centro_y + radio * np.sin(angulos)).astype(int)

    # Solo contamos los puntos que caen dentro de la imagen
    dentro     = (xi_muestra >= 0) & (xi_muestra < ancho) & (yi_muestra >= 0) & (yi_muestra < alto)
    xi_validos = xi_muestra[dentro]
    yi_validos = yi_muestra[dentro]

    if len(xi_validos) == 0:
        return 0.0

    bordes = rejilla_bordes[yi_validos, xi_validos]

    puntos_con_borde = int(np.sum(bordes))

    # ── CÁLCULO DE APTITUD (único, no duplicado como en el original) ──
    aptitud_base = puntos_con_borde / max(1, len(xi_validos))

    # Penalización por fragmentación: bordes dispersos vs. arco continuo
    # Duplicamos para manejar wrap-around del círculo
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

    # Combinación: 60% cobertura total + 40% continuidad
    aptitud = aptitud_base * (0.60 + 0.40 * fraccion_consecutiva)

    # Penalizamos los círculos muy pequeños para evitar detectar ruido
    if radio < RADIO_MINIMO:
        aptitud = aptitud * (radio / RADIO_MINIMO)

    return float(aptitud)


# ── Verificar que el círculo sea realmente circular ───────────────────────────

def verificar_continuidad_circulo(centro_x, centro_y, radio, rejilla_bordes, forma_imagen,
                                   num_sectores=NUM_SECTORES_VALIDACION):
    """
    Confirma que los bordes que respaldan el círculo estén distribuidos de forma
    continua alrededor de la circunferencia, no dispersos en puntitos sueltos.

    Dividimos el círculo en partes iguales (como una pizza en 24 porciones) y
    revisamos cuáles tienen borde. Luego calculamos tres cosas:
      - fraccion   : qué proporción del total de partes tiene borde.
      - max_consecutivos : cuántas partes seguidas con borde hay como máximo.
      - numero_arcos : en cuántos grupos separados están esas partes con borde.

    Un círculo real, aunque esté parcialmente tapado, cumple las tres.
    Un polígono falla al menos una: sus lados solo cruzan el círculo
    en puntos aislados, sin formar arcos continuos.

    Retorna (fraccion, max_consecutivos, numero_arcos).
    """
    alto, ancho = forma_imagen[:2]
    sectores    = []  # True si esa porción del círculo tiene borde

    for s in range(num_sectores):
        angulo_inicio = 2.0 * math.pi * s       / num_sectores
        angulo_fin    = 2.0 * math.pi * (s + 1) / num_sectores
        hay_borde     = False

        for paso in range(PUNTOS_POR_SECTOR):
            t      = paso / PUNTOS_POR_SECTOR
            angulo = angulo_inicio + (angulo_fin - angulo_inicio) * t
            xi     = int(round(centro_x + radio * math.cos(angulo)))
            yi     = int(round(centro_y + radio * math.sin(angulo)))

            if 0 <= xi < ancho and 0 <= yi < alto:
                if rejilla_bordes[yi, xi]:
                    hay_borde = True
                    break

        sectores.append(hay_borde)

    # Contamos cuántas partes tienen borde
    total_con_borde = sum(1 for tiene in sectores if tiene)
    fraccion = total_con_borde / num_sectores

    # Buscamos la racha más larga de partes seguidas con borde
    # Duplicamos la lista para manejar el wrap del círculo (que no tiene inicio ni fin)
    secuencia        = sectores + sectores
    max_consecutivos = 0
    consecutivos     = 0
    for tiene in secuencia:
        if tiene:
            consecutivos += 1
            if consecutivos > max_consecutivos:
                max_consecutivos = consecutivos
        else:
            consecutivos = 0
    if max_consecutivos > num_sectores:
        max_consecutivos = num_sectores

    # Contamos cuántos grupos separados de borde hay
    numero_arcos = 0
    if total_con_borde > 0:
        for i in range(num_sectores):
            sector_prev = sectores[(i - 1) % num_sectores]
            sector_curr = sectores[i]
            if sector_curr and not sector_prev:
                numero_arcos += 1

    return fraccion, max_consecutivos, numero_arcos
