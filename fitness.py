"""
Función de aptitud basada en Ayala-Ramírez et al. (2006), Sección 3.
Un individuo codifica 3 índices de puntos de borde. A partir de esos
3 puntos se determina el círculo por circunscripción y se evalúa
cuántos píxeles de borde caen cerca de ese círculo.
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


def evaluate_fitness(individual: np.ndarray, edge_points: np.ndarray,
                     img_shape: tuple, delta: float = 2.0) -> float:
    """
    Aptitud = fracción de puntos de borde que están a distancia <= delta
    del círculo candidato.  Eq. (1) del paper.

    individual: array de 3 índices enteros en edge_points
    delta: tolerancia en píxeles (paper usa ~2px)
    """
    i0, i1, i2 = individual
    p1, p2, p3 = edge_points[i0], edge_points[i1], edge_points[i2]
    result = circle_from_three_points(p1, p2, p3)
    if result is None:
        return 0.0

    cx, cy, r = result
    h, w = img_shape[:2]
    if r < 5 or r > min(h, w) / 2:
        return 0.0
    if not (0 <= cx < w and 0 <= cy < h):
        return 0.0

    dists = np.abs(np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2) - r)
    fitness = np.sum(dists <= delta) / len(edge_points)
    return float(fitness)
