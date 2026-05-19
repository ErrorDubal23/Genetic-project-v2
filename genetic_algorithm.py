"""
GA mejorado para detección de círculos.
Basado en Ayala-Ramírez et al. (2006) + mejoras de robustez.

CAMBIOS CLAVE respecto a v1 (original):
1. Distance Transform precomputado: fitness 100x más rápida que cKDTree.
2. Fitness híbrida (circunferencia + inliers): precisión del paper sin fragilidad.
3. Canonicalización de cromosomas: reduce espacio de búsqueda (simetría i,j,k).
4. NMS con AND lógico (posición Y radio): evita duplicados sin matar círculos cercanos.
5. top_k=1 por defecto: evita falsos positivos de detección múltiple.
6. Elitismo eficiente: evalúa offspring una sola vez por generación.

CAMBIOS CLAVE respecto a v2 (fallido):
- Quitado min_circumference_ratio DURO que eliminaba círculos válidos.
- Quitado cKDTree que hacía cada imagen tardar minutos.
- Quitado occlusion_penalty mal diseñado que destruía fitness en bordes de imagen.
- Radio mínimo vuelve a 5.0 (como original), no 8.0.
- delta por defecto 2.0 (como original), no 1.5.
- Fitness híbrida: no depende solo del muestreo de circunferencia.
"""
import numpy as np
from fitness import evaluate_fitness, circle_from_three_points
from scipy import ndimage


class GeneticCircleDetector:
    def __init__(
        self,
        population_size: int = 70,
        crossover_prob: float = 0.55,
        mutation_prob: float = 0.10,
        elite_count: int = 2,
        max_generations: int = 500,
        min_radius: float = 5.0,
        max_radius: float = None,
        fitness_delta: float = 2.0,
        nms_threshold: float = 15.0,
        top_k: int = 1,
        fitness_threshold: float = 0.10,
    ):
        self.pop_size = population_size
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.elite = elite_count
        self.max_gen = max_generations
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.fitness_delta = fitness_delta
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.fitness_threshold = fitness_threshold

    # ------------------------------------------------------------------
    def _init_population(self, n_points: int) -> np.ndarray:
        """Población sin índices repetidos por individuo."""
        pop = np.zeros((self.pop_size, 3), dtype=np.int64)
        for i in range(self.pop_size):
            pop[i] = np.random.choice(n_points, size=3, replace=False)
        return pop

    def _canonicalize(self, pop):
        """
        Ordena los 3 genes de menor a mayor.
        (i,j,k), (j,i,k), (k,j,i) codifican el MISMO círculo.
        Ordenar elimina esta simetría y reduce el espacio de búsqueda.
        """
        return np.sort(pop, axis=1)

    def _evaluate(self, pop, edge_points, img_shape, dt):
        return np.array([
            evaluate_fitness(
                ind, edge_points, img_shape, dt,
                delta=self.fitness_delta,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
            )
            for ind in pop
        ])

    def _roulette_select(self, pop, fitnesses):
        total = fitnesses.sum()
        if total <= 1e-12:
            probs = np.ones(len(pop)) / len(pop)
        else:
            probs = fitnesses / total
        idx = np.random.choice(len(pop), size=len(pop), replace=True, p=probs)
        return pop[idx]

    def _crossover(self, pop):
        new_pop = pop.copy()
        for i in range(0, self.pop_size - 1, 2):
            if np.random.rand() < self.pc:
                point = np.random.randint(1, 3)
                new_pop[i, point:], new_pop[i + 1, point:] = (
                    pop[i + 1, point:].copy(),
                    pop[i, point:].copy(),
                )
        return self._canonicalize(new_pop)

    def _mutate(self, pop, n_points):
        for i in range(self.pop_size):
            if np.random.rand() < self.pm:
                gene = np.random.randint(3)
                pop[i, gene] = np.random.randint(n_points)
        return self._canonicalize(pop)

    def _non_max_suppression(self, candidates):
        """
        Supresión no-máxima: dos círculos son duplicados solo si son similares
        en posición Y radio simultáneamente (AND lógico).
        Esto evita que círculos reales cercanos sean fusionados por error.
        """
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda x: x["fitness"], reverse=True)
        keep = []
        for c in candidates:
            overlap = False
            for k in keep:
                d = np.hypot(c["x"] - k["x"], c["y"] - k["y"])
                dr = abs(c["r"] - k["r"])
                if d < self.nms_threshold and dr < (self.nms_threshold / 2):
                    overlap = True
                    break
            if not overlap:
                keep.append(c)
                if len(keep) >= self.top_k:
                    break
        return keep

    # ------------------------------------------------------------------
    def detect(self, edge_points: np.ndarray, img_shape: tuple,
               edge_map: np.ndarray = None) -> dict:
        n = len(edge_points)
        if n < 3:
            return {"circles": [], "best_fitness": 0.0}

        # Precomputar Distance Transform UNA SOLA VEZ por imagen.
        # dt[y, x] = distancia del píxel (y,x) al borde más cercano.
        # Esto reemplaza cKDTree (lento) con lookup O(1) vectorizado.
        if edge_map is None:
            # Fallback: reconstruir edge_map desde edge_points
            edge_map = np.zeros(img_shape[:2], dtype=np.uint8)
            if n > 0:
                xi = np.clip(edge_points[:, 0].astype(int), 0, img_shape[1] - 1)
                yi = np.clip(edge_points[:, 1].astype(int), 0, img_shape[0] - 1)
                edge_map[yi, xi] = 255
        dt = ndimage.distance_transform_edt(edge_map == 0)

        pop = self._init_population(n)
        best_ind, best_fit = None, -1.0

        for _ in range(self.max_gen):
            fits = self._evaluate(pop, edge_points, img_shape, dt)

            # Elitismo
            elite_idx = np.argsort(fits)[-self.elite:]
            elites = pop[elite_idx].copy()

            gen_best = elite_idx[-1]
            if fits[gen_best] > best_fit:
                best_fit = fits[gen_best]
                best_ind = pop[gen_best].copy()

            # Reproducción
            selected = self._roulette_select(pop, fits)
            offspring = self._crossover(selected)
            offspring = self._mutate(offspring, n)

            # Reemplazar peores offspring con élites (evaluar offspring 1 sola vez)
            offspring_fits = self._evaluate(offspring, edge_points, img_shape, dt)
            worst_idx = np.argsort(offspring_fits)[:self.elite]
            for k, wi in enumerate(worst_idx):
                offspring[wi] = elites[k]

            pop = self._canonicalize(offspring)

        # Extraer múltiples círculos candidatos de la población final
        circles = []
        final_fits = self._evaluate(pop, edge_points, img_shape, dt)
        seen = set()

        for idx in np.argsort(final_fits)[::-1]:
            fit = final_fits[idx]
            if fit < self.fitness_threshold:
                break
            ind = tuple(pop[idx])
            if ind in seen:
                continue
            seen.add(ind)

            p1, p2, p3 = edge_points[ind[0]], edge_points[ind[1]], edge_points[ind[2]]
            result = circle_from_three_points(p1, p2, p3)
            if result is None:
                continue
            cx, cy, r = result

            circles.append({
                "x": round(float(cx), 2),
                "y": round(float(cy), 2),
                "r": round(float(r), 2),
                "fitness": round(float(fit), 4),
            })

            if len(circles) >= self.top_k * 3:
                break

        circles = self._non_max_suppression(circles)

        return {
            "circles": circles,
            "best_fitness": round(float(best_fit), 4) if best_fit > 0 else 0.0,
        }
