"""
GA para detección de círculos – mejoras conservadoras sobre el código original.

MEJORAS:
1. Detección multi-círculo: extrae círculos secuenciales eliminando inliers
   del círculo detectado, como propone el paper (Sección 3.1.4).
2. Cromosomas sin repetición: evita individuos degenerados como (5,5,8).
3. Canonicalización: ordena i≤j≤k para que (3,5,8)=(5,3,8), reduciendo
   el espacio de búsqueda ~6×.
4. Evaluación de offspring 1 sola vez por generación (antes se evaluaba 2×).
"""
import numpy as np
from fitness import evaluate_fitness, circle_from_three_points


class GeneticCircleDetector:
    def __init__(
        self,
        population_size: int = 70,
        crossover_prob: float = 0.55,
        mutation_prob: float = 0.10,
        elite_count: int = 2,
        max_generations: int = 500,
        top_k: int = 5,
        fitness_threshold: float = 0.01,
    ):
        self.pop_size = population_size
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.elite = elite_count
        self.max_gen = max_generations
        self.top_k = top_k
        self.fitness_threshold = fitness_threshold

    # ------------------------------------------------------------------
    def _init_population(self, n_points: int) -> np.ndarray:
        """Población donde cada individuo tiene 3 índices distintos."""
        pop = np.zeros((self.pop_size, 3), dtype=np.int64)
        for i in range(self.pop_size):
            pop[i] = np.random.choice(n_points, size=3, replace=False)
        return pop

    def _canonicalize(self, pop):
        """Ordena genes i≤j≤k para eliminar simetría de permutación."""
        return np.sort(pop, axis=1)

    def _evaluate(self, pop, edge_points, img_shape, delta):
        return np.array([evaluate_fitness(ind, edge_points, img_shape, delta) for ind in pop])

    def _roulette_select(self, pop, fitnesses):
        total = fitnesses.sum()
        if total == 0:
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
        """Mutación que evita repetir índices dentro del mismo individuo."""
        for i in range(self.pop_size):
            if np.random.rand() < self.pm:
                gene = np.random.randint(3)
                existing = set(pop[i])
                candidates = [c for c in range(n_points) if c not in existing]
                if candidates:
                    pop[i, gene] = np.random.choice(candidates)
                else:
                    pop[i, gene] = np.random.randint(n_points)
        return self._canonicalize(pop)

    def _nms(self, candidates):
        """Supresión no-máxima: conserva círculos no superpuestos en posición+radio."""
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda x: x["fitness"], reverse=True)
        keep = []
        for c in candidates:
            dup = False
            for k in keep:
                d = np.hypot(c["x"] - k["x"], c["y"] - k["y"])
                if d < 15 or abs(c["r"] - k["r"]) < 10:
                    dup = True
                    break
            if not dup:
                keep.append(c)
                if len(keep) >= self.top_k:
                    break
        return keep

    # ------------------------------------------------------------------
    def _run_ga(self, edge_points, img_shape, delta):
        """Una ejecución del GA. Devuelve (círculo_mejor, fitness_mejor)."""
        n = len(edge_points)
        if n < 3:
            return None, 0.0

        pop = self._init_population(n)
        best_ind, best_fit = None, -1.0

        for _ in range(self.max_gen):
            fits = self._evaluate(pop, edge_points, img_shape, delta)

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
            off_fits = self._evaluate(offspring, edge_points, img_shape, delta)
            worst_idx = np.argsort(off_fits)[:self.elite]
            for k, wi in enumerate(worst_idx):
                offspring[wi] = elites[k]

            pop = self._canonicalize(offspring)

        if best_ind is None or best_fit <= self.fitness_threshold:
            return None, best_fit

        p1, p2, p3 = edge_points[best_ind[0]], edge_points[best_ind[1]], edge_points[best_ind[2]]
        result = circle_from_three_points(p1, p2, p3)
        if result is None:
            return None, best_fit

        cx, cy, r = result
        return {"x": round(float(cx), 2), "y": round(float(cy), 2),
                "r": round(float(r), 2), "fitness": round(float(best_fit), 4)}, best_fit

    def detect(self, edge_points: np.ndarray, img_shape: tuple,
                delta: float = 2.0) -> dict:
        """
        Detección multi-círculo:
        1. Ejecuta GA, extrae mejor círculo.
        2. Elimina edge_points que son inliers de ese círculo (los 'consume').
        3. Repite mientras queden ≥3 puntos y fitness > umbral.
        4. Aplica NMS por si quedaron duplicados.
        """
        remaining = edge_points.copy()
        all_circles = []
        global_best = 0.0

        for _ in range(self.top_k):
            if len(remaining) < 3:
                break

            circle, fit = self._run_ga(remaining, img_shape, delta)
            if circle is None or fit <= self.fitness_threshold:
                break

            all_circles.append(circle)
            if fit > global_best:
                global_best = fit

            # Eliminar inliers del círculo detectado para buscar el siguiente
            cx, cy, r = circle["x"], circle["y"], circle["r"]
            dists = np.abs(np.sqrt((remaining[:, 0] - cx)**2 + (remaining[:, 1] - cy)**2) - r)
            inliers = dists <= delta * 2.5  # margen generoso para no fragmentar círculos
            remaining = remaining[~inliers]

        all_circles = self._nms(all_circles)
        return {"circles": all_circles, "best_fitness": round(float(global_best), 4)}
