from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import numpy as np

from image_utils import load_image_from_bytes, preprocess, get_edge_points, annotate_image, ndarray_to_b64
from genetic_algorithm import GeneticCircleDetector

app = FastAPI(title="CircleGA API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    params: str = Form(default="{}"),
):
    try:
        p = json.loads(params)
    except Exception:
        raise HTTPException(400, "params debe ser JSON válido")

    data = await image.read()
    try:
        img = load_image_from_bytes(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    edge = preprocess(img)
    edge_points = get_edge_points(edge)

    if len(edge_points) < 3:
        raise HTTPException(422, "La imagen no tiene suficientes bordes detectados")

    detector = GeneticCircleDetector(
        population_size=int(p.get("population_size", 70)),
        crossover_prob=float(p.get("crossover_prob", 0.55)),
        mutation_prob=float(p.get("mutation_prob", 0.10)),
        elite_count=int(p.get("elite_count", 2)),
        max_generations=int(p.get("max_generations", 500)),
        min_radius=float(p.get("min_radius", 8.0)),
        max_radius=float(p.get("max_radius", None)),
        fitness_delta=float(p.get("fitness_delta", 1.5)),
        min_circumference_ratio=float(p.get("min_circumference_ratio", 0.55)),
        nms_threshold=float(p.get("nms_threshold", 15.0)),
        top_k=int(p.get("top_k", 5)),
        occlusion_penalty=float(p.get("occlusion_penalty", 0.15)),
    )

    result = detector.detect(edge_points, img.shape)

    annotated = annotate_image(img, result["circles"])
    img_b64 = ndarray_to_b64(annotated)

    # Métrica de calidad: error promedio de inliers respecto al radio
    avg_error = 0.0
    if result["circles"]:
        c = result["circles"][0]
        dists = np.sqrt((edge_points[:, 0] - c["x"])**2 + (edge_points[:, 1] - c["y"])**2)
        inliers = np.abs(dists - c["r"]) <= c.get("delta", 1.5)
        if inliers.sum() > 0:
            avg_error = float(np.mean(np.abs(dists[inliers] - c["r"])))

    return JSONResponse({
        "circles": result["circles"],
        "count": len(result["circles"]),
        "avg_error": round(avg_error, 4),
        "fitness": result["best_fitness"],
        "annotated_image_b64": img_b64,
    })
