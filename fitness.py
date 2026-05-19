"""
Función de aptitud mejorada basada en Ayala-Ramírez et al. (2006).
Mejoras:
1. Fitness basado en muestreo de circunferencia (como el paper original)
2. Ratio de circunferencia para descartar falsos positivos
3. Penalización por radio pequeño y oclusión
4. Validación de puntos colineales y radios extremos
"""
import numpy as np


def circle_from_three_points(p1, p2, p3):
    """
    Devuelve (cx, cy, r) del círculo que pasa por tres puntos.
    Retorna None si los puntos son colineales o muy cercanos a serlo.
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
    delta: float = 1.5,
    min_radius: float = 8.0,
    max_radius: float = None,
    min_circumference_ratio: float = 0.55,
    occlusion_penalty: float = 0.15,
) -> float:
    """
    MEJORA: Aptitud basada en muestreo de circunferencia (como Eq. 6-8 del paper).
    En lugar de contar bordes cercanos al círculo, muestreamos puntos sobre
    la circunferencia y verificamos si hay bordes reales ahí.

    Además:
    - Penaliza radios pequeños (< min_radius) para evitar ruido puntual
    - Exige un mínimo ratio de circunferencia cubierta (descarta falsos positivos)
    - Penaliza oclusión parcial moderadamente
    """
    i0, i1, i2 = individual
    p1, p2, p3 = edge_points[i0], edge_points[i1], edge_points[i2]
    result = circle_from_three_points(p1, p2, p3)
    if result is None:
        return 0.0

    cx, cy, r = result
    h, w = img_shape[:2]

    # Radio inválido
    if r < min_radius:
        return 0.0
    if max_radius is not None and r > max_radius:
        return 0.0
    if r > min(h, w) / 2:
        return 0.0

    # Centro fuera de la imagen con margen
    margin = r + delta
    if not (-margin <= cx < w + margin and -margin <= cy < h + margin):
        return 0.0

    # --- Muestreo de circunferencia (Eq. 6-8 del paper) ---
    # Ns = número de puntos a muestrear = perímetro aproximado
    Ns = max(20, int(2 * np.pi * r))
    angles = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)

    # Solo considerar puntos dentro de la imagen
    valid_mask = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)
    xs_valid = xs[valid_mask]
    ys_valid = ys[valid_mask]

    if len(xs_valid) == 0:
        return 0.0

    # Buscar bordes cercanos a cada punto muestreado
    # Usamos KD-tree para búsqueda eficiente de vecinos
    from scipy.spatial import cKDTree
    tree = cKDTree(edge_points)
    distances, _ = tree.query(np.column_stack((xs_valid, ys_valid)), k=1)

    # Puntos de circunferencia con borde cercano
    matched = np.sum(distances <= delta)

    # Ratio de circunferencia cubierta
    circumference_ratio = matched / Ns

    # Penalización por radio pequeño (Eq. 9 del paper, adaptada)
    radius_penalty = 1.0
    if r < 15:
        radius_penalty = r / 15.0

    # Penalización por oclusión (parte de la circunferencia sin borde)
    visible_ratio = matched / len(xs_valid) if len(xs_valid) > 0 else 0
    occlusion_factor = 1.0 - (occlusion_penalty * (1.0 - visible_ratio))
    occlusion_factor = max(0.5, occlusion_factor)

    # Fitness final
    fitness = circumference_ratio * radius_penalty * occlusion_factor

    # Descartar si no cumple ratio mínimo de circunferencia (evita falsos positivos)
    if circumference_ratio < min_circumference_ratio:
        fitness *= 0.1

    return float(fitness)
