"""
Función de aptitud HÍBRIDA:
- 65% muestreo de circunferencia (fiel al paper, Eq. 6-8 de Ayala-Ramírez)
- 35% inliers globales (robustez heredada del código original)

MEJORAS respecto a v1 (código original):
- Distance Transform precomputado elimina cKDTree lento y permite O(1) por punto.
- Híbrido: evita que círculos gigantes aleatorios con muchos inliers espurios
  tengan fitness alta (problema del original), y evita que círculos reales
  con bordes gruesos/debilidos sean descartados (problema de v2 estricto).
- Penalización suave por cobertura baja en vez de umbral duro que eliminaba
  círculos válidos con oclusión parcial o bordes irregulares.
- Penalización de radio pequeño según Eq. 9 del paper (suave, no eliminatoria).
"""
import numpy as np


def circle_from_three_points(p1, p2, p3):
    """
    Devuelve (cx, cy, r) del círculo que pasa por tres puntos.
    Retorna None si los puntos son colineales.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = np.sqrt((ax - ux)**2 + (ay - uy)**2)
    return ux, uy, r


def evaluate_fitness(
    individual: np.ndarray,
    edge_points: np.ndarray,
    img_shape: tuple,
    dt: np.ndarray,
    delta: float = 2.0,
    min_radius: float = 5.0,
    max_radius: float = None,
) -> float:
    """
    individual: array de 3 índices enteros en edge_points
    dt: Distance Transform precomputado (distancia de cada píxel al borde más cercano)
    delta: tolerancia en píxeles para considerar un punto como "sobre el borde"
    """
    i0, i1, i2 = individual
    p1, p2, p3 = edge_points[i0], edge_points[i1], edge_points[i2]
    result = circle_from_three_points(p1, p2, p3)
    if result is None:
        return 0.0

    cx, cy, r = result
    h, w = img_shape[:2]

    # Validaciones básicas (igual que el original)
    if r < min_radius:
        return 0.0
    if max_radius is not None and r > max_radius:
        return 0.0
    if r > min(h, w) / 2:
        return 0.0

    margin = r + delta
    if not (-margin <= cx < w + margin and -margin <= cy < h + margin):
        return 0.0

    # ------------------------------------------------------------------
    # COMPONENTE 1: Muestreo de circunferencia (paper, Eq. 6-8)
    # ------------------------------------------------------------------
    # Ns ~ perímetro. Más radio = más puntos de muestreo para mantener resolución.
    Ns = max(20, int(2 * np.pi * r))
    angles = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)

    # Puntos de muestreo dentro de la imagen
    valid_mask = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)
    if not valid_mask.any():
        return 0.0

    xs_v = xs[valid_mask]
    ys_v = ys[valid_mask]
    # Redondeo al píxel más cercano para consultar dt
    xi = np.clip(np.rint(xs_v).astype(int), 0, w - 1)
    yi = np.clip(np.rint(ys_v).astype(int), 0, h - 1)

    # dt[yi, xi] = distancia euclidiana al borde más cercano
    dists_sampled = dt[yi, xi]
    matched = np.sum(dists_sampled <= delta)
    # Normalizar por Ns TOTAL (puntos fuera de imagen cuentan como no-matched)
    circum_ratio = matched / Ns

    # ------------------------------------------------------------------
    # COMPONENTE 2: Inliers globales (robustez del código original)
    # ------------------------------------------------------------------
    # Proporción de TODOS los edge_points que caen cerca de la circunferencia.
    # Normalizado por perímetro para que círculos gigantes no ganen solo por
    # absorber ruido aleatorio (falso positivo clásico del enfoque original).
    dists_to_circ = np.abs(
        np.sqrt((edge_points[:, 0] - cx) ** 2 + (edge_points[:, 1] - cy) ** 2) - r
    )
    inliers = np.sum(dists_to_circ <= delta)
    # El máximo teórico de inliers útiles es ~perímetro. Cap en 1.0.
    inlier_ratio = min(inliers / (2.2 * np.pi * r), 1.0) if r > 0 else 0.0

    # ------------------------------------------------------------------
    # COMBINACIÓN HÍBRIDA
    # ------------------------------------------------------------------
    # 65% circunferencia (precisión del paper) + 35% inliers (robustez original)
    fitness = 0.65 * circum_ratio + 0.35 * inlier_ratio

    # Penalización por radio pequeño (Eq. 9 del paper adaptada)
    # Suave: no elimina, solo reduce. Evita que puntos de ruido sean detectados.
    if r < 15:
        fitness *= (r / 15.0)

    # Penalización SUAVE por cobertura muy baja.
    # En v2 había un umbral DURO (0.55) que eliminaba círculos válidos.
    # Ahora es gradual: cobertura baja reduce fitness pero no mata al individuo.
    if circum_ratio < 0.15:
        fitness *= 0.2
    elif circum_ratio < 0.30:
        fitness *= 0.5
    elif circum_ratio < 0.45:
        fitness *= 0.8

    return float(fitness)
