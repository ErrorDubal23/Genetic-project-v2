"""
Función de aptitud mejorada sobre la base original.

MEJORAS respecto al código original:
1. Normalización por perímetro (no por total de edge_points).
   El original hacía: fitness = inliers / len(edge_points).
   Problema: círculos gigantes ganan automáticamente porque su circunferencia
   grande atraviesa más zonas con puntos de borde aleatorios (falsos positivos).
   Ahora: fitness = inliers / (2πr), midiendo qué fracción del perímetro
   teórico está realmente cubierto por bordes.

2. Verificación de distribución angular.
   El original aceptaba un arco de 90° con 20 inliers como "círculo".
   Ahora: dividimos la circunferencia en 8 sectores. Si inliers caen en
   menos de 5 sectores, es un arco parcial → penalización fuerte.
   Esto elimina falsos positivos de líneas curvas o esquinas redondeadas.

3. Penalización por radio pequeño (Eq. 9 del paper, suave).
   Radios < 15px son más propensos a ser ruido o artefactos.
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
    MEJORADO sobre el original:
    - Normalización por perímetro (evita favoritismo a círculos gigantes)
    - Verificación angular (evita arcos parciales como falsos positivos)
    - Penalización suave por radio pequeño
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

    # Distancia de cada edge_point a la circunferencia candidato
    dists = np.abs(np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2) - r)
    inlier_mask = dists <= delta
    n_inliers = np.sum(inlier_mask)

    if n_inliers == 0:
        return 0.0

    # ------------------------------------------------------------------
    # MEJORA 1: Circumference ratio (inliers / perímetro teórico)
    # El original dividía por len(edge_points), favoreciendo círculos gigantes.
    # Ahora medimos: ¿qué fracción del borde del círculo está presente?
    perimeter = 2.0 * np.pi * r
    circumference_ratio = min(n_inliers / (perimeter * 0.9), 1.0)
    # Factor 0.9: un círculo perfecto en mapa de 1px tiene ~perímetro/0.9 inliers
    # porque los bordes están en píxeles discretos.

    # ------------------------------------------------------------------
    # MEJORA 2: Distribución angular (descarta arcos parciales)
    # Dividimos [0, 2π) en 8 sectores. Contamos cuántos tienen al menos 1 inlier.
    inlier_points = edge_points[inlier_mask]
    angles = np.arctan2(inlier_points[:, 1] - cy, inlier_points[:, 0] - cx)
    angles = np.mod(angles + 2 * np.pi, 2 * np.pi)  # [0, 2π)
    sectors = (angles / (np.pi / 4)).astype(int) % 8
    occupied_sectors = len(np.unique(sectors))

    # Si inliers están en menos de 5 de 8 sectores, es un arco, no círculo
    if occupied_sectors < 5:
        angular_penalty = 0.2  # Penalización fuerte: arco parcial
    elif occupied_sectors < 7:
        angular_penalty = 0.7  # Penalización moderada
    else:
        angular_penalty = 1.0  # Distribución completa → círculo real

    # ------------------------------------------------------------------
    # MEJORA 3: Penalización por radio pequeño (Eq. 9 del paper, suave)
    if r < 15:
        radius_penalty = r / 15.0
    else:
        radius_penalty = 1.0

    # ------------------------------------------------------------------
    # Fitness combinada
    # Peso: 70% cobertura de circunferencia + 30% distribución angular
    fitness = (0.70 * circumference_ratio + 0.30 * angular_penalty) * radius_penalty

    return float(fitness)
