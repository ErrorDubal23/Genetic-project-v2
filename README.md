# Genetic-project-v2

Mejora de detección de círculos con Algoritmos Genéticos, basado en Ayala-Ramírez et al. (2006).

## Estructura

- `main.py` — API FastAPI
- `genetic_algorithm.py` — GA con mejoras (NMS, canonicalización, multi-círculo)
- `fitness.py` — Función de aptitud por muestreo de circunferencia
- `image_utils.py` — Preprocesado con CLAHE + Canny + dilatación ligera

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
uvicorn main:app --reload --port 8000
```

## Endpoints

- `GET /health` — Healthcheck
- `POST /detect` — Subir imagen y detectar círculos

### Parámetros opcionales (POST /detect, campo `params` como JSON string):

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| population_size | 70 | Tamaño de población |
| crossover_prob | 0.55 | Probabilidad de cruce |
| mutation_prob | 0.10 | Probabilidad de mutación |
| elite_count | 2 | Individuos élites |
| max_generations | 500 | Máximo de generaciones |
| min_radius | 8.0 | Radio mínimo permitido |
| max_radius | null | Radio máximo (null = auto) |
| fitness_delta | 1.5 | Tolerancia de borde en píxeles |
| min_circumference_ratio | 0.55 | Ratio mínimo de circunferencia cubierta |
| nms_threshold | 15.0 | Umbral de supresión no-máxima |
| top_k | 5 | Máximo de círculos a retornar |
| occlusion_penalty | 0.15 | Penalización por oclusión |

## Mejoras respecto a v1

1. **Canonicalización de cromosomas** — ordena genes para reducir espacio de búsqueda
2. **Fitness por muestreo de circunferencia** — fiel al paper original, más preciso
3. **Ratio de circunferencia** — descarta falsos positivos con cobertura baja
4. **NMS + top-k** — detecta múltiples círculos sin duplicados
5. **Preprocesado robusto** — CLAHE + Canny + dilatación para bordes débiles y ruido
6. **Penalización por oclusión** — tolera círculos parcialmente ocluidos
