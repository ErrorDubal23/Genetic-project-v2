"""
Implementación mejorada del GA para detección de círculos.
Basado en Ayala-Ramírez et al. (2006), con mejoras:
1. Detección multi-círculo con NMS y top-k
2. Fitness basado en muestreo de circunferencia (como el paper)
3. Penalización por radio pequeño y oclusión
4. Reordenamiento de genes para eliminar simetría en cromosomas
5. Validación estricta de individuos para reducir falsos positivos
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
        min_radius: float = 8.0,
        max_radius: float = None,
        fitness_delta: float = 1.5,
        min_circumference_ratio: float = 0.55,
        nms_threshold: float = 15.0,
        top_k: int = 5,
        occlusion_penalty: float = 0.15,
    ):
        self.pop_size = population_size
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.elite = elite_count
        self.max_gen = max_generations
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.fitness_delta = fitness_delta
        self.min_circumference_ratio = min_circumference_ratio
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.occlusion_penalty = occlusion_penalty

    # ------------------------------------------------------------------
    def _init_population(self, n_points: int) -> np.ndarray:
        """Genera población única: filtra índices repetidos por individuo."""
        pop = np.zeros((self.pop_size, 3), dtype=np.int64)
        for i in range(self.pop_size):
            pop[i] = np.random.choice(n_points, size=3, replace=False)
        return pop

    def _canonicalize(self, pop):
        """
        MEJORA 1: Ordena los 3 genes de menor a mayor.
        Los genes (índices de puntos de borde) son intercambiables:
        (i,j,k), (j,i,k), (k,j,i) codifican el MISMO círculo.
        Ordenarlos reduce el espacio de búsqueda efectivo y evita
        que el GA gaste evaluaciones en individuos equivalentes.
        """
        return np.sort(pop, axis=1)

    def _evaluate(self, pop, edge_points, img_shape):
        return np.array([
            evaluate_fitness(
                ind, edge_points, img_shape,
                delta=self.fitness_delta,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                min_circumference_ratio=self.min_circumference_ratio,
                occlusion_penalty=self.occlusion_penalty,
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
                # MEJORA 2: Mutación inteligente - evita colisiones
                existing = set(pop[i])
                candidates = [c for c in range(n_points) if c not in existing]
                if candidates:
                    pop[i, gene] = np.random.choice(candidates)
                else:
                    pop[i, gene] = np.random.randint(n_points)
        return self._canonicalize(pop)

    def _non_max_suppression(self, candidates):
        """
        MEJORA 3: NMS para multi-círculo.
        Filtra círculos superpuestos quedándose con el de mayor fitness.
        """
        if not candidates:
            return []
        # Ordenar por fitness descendente
        candidates = sorted(candidates, key=lambda x: x["fitness"], reverse=True)
        keep = []
        for c in candidates:
            overlap = False
            for k in keep:
                d = np.hypot(c["x"] - k["x"], c["y"] - k["y"])
                if d < self.nms_threshold or abs(c["r"] - k["r"]) < self.nms_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(c)
                if len(keep) >= self.top_k:
                    break
        return keep

    # ------------------------------------------------------------------
    def detect(self, edge_points: np.ndarray, img_shape: tuple) -> dict:
        n = len(edge_points)
        if n < 3:
            return {"circles": [], "best_fitness": 0.0}

        pop = self._init_population(n)
        best_ind, best_fit = None, -1.0
        fitness_history = []

        for gen in range(self.max_gen):
            fits = self._evaluate(pop, edge_points, img_shape)
            fitness_history.append(float(fits.max()))

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

            # Reemplazar los peores con los élites
            worst_idx = np.argsort(
                self._evaluate(offspring, edge_points, img_shape)
            )[: self.elite]
            for k, wi in enumerate(worst_idx):
                offspring[wi] = elites[k]

            pop = self._canonicalize(offspring)

            # Criterio de parada temprana si converge
            if gen > 50 and np.std(fitness_history[-30:]) < 1e-4:
                break

        # Extraer múltiples círculos de la población final
        circles = []
        final_fits = self._evaluate(pop, edge_points, img_shape)
        seen = set()

        for idx in np.argsort(final_fits)[::-1]:
            if final_fits[idx] <= 0:
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
                "fitness": round(float(final_fits[idx]), 4),
                "delta": self.fitness_delta,
            })

            if len(circles) >= self.top_k * 3:  # candidatos extra para NMS
                break

        # Aplicar NMS y devolver top_k
        circles = self._non_max_suppression(circles)

        return {
            "circles": circles,
            "best_fitness": round(float(best_fit), 4) if best_fit > 0 else 0.0,
        }
