"""
Este es el punto de entrada de la API. Recibe la imagen desde el frontend,
la procesa con el algoritmo genético y devuelve los círculos detectados.

MEJORA: el error promedio ahora se calcula solo sobre los inliers del círculo
principal, no sobre todos los puntos de borde. Esto da una métrica realista
cuando hay múltiples círculos u otros objetos en la imagen.
"""

import json
import os
import numpy as np

ORIGENES_PERMITIDOS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ARCHIVOS_DEL_PROYECTO = {
    "genetic_algorithm.py",
    "fitness.py",
    "image_utils.py",
    "main.py",
}

DIRECTORIO_BACKEND = os.path.dirname(os.path.abspath(__file__))

from image_utils import (
    cargar_imagen_desde_bytes,
    preprocesar,
    obtener_puntos_borde,
    anotar_imagen,
    imagen_a_base64,
)
from genetic_algorithm import (
    DetectorCirculosGA,
    TAMANIO_POBLACION,
    PROB_CRUCE,
    PROB_MUTACION,
    NUM_ELITE,
    MAX_GENERACIONES,
    DELTA_TOLERANCIA,
)

# ── Configuración de la aplicación ────────────────────────────────────────────

aplicacion = FastAPI(
    title="CircleGA API",
    description="Detección de círculos en imágenes usando algoritmos genéticos.",
    version="2.0.0",
)

aplicacion.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_PERMITIDOS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@aplicacion.get("/health")
def verificar_estado():
    return {"estado": "ok"}


@aplicacion.get("/codigo/{nombre_archivo}")
def obtener_codigo_fuente(nombre_archivo: str):
    if nombre_archivo not in ARCHIVOS_DEL_PROYECTO:
        raise HTTPException(status_code=404, detail="Archivo no permitido.")

    ruta = os.path.join(DIRECTORIO_BACKEND, nombre_archivo)

    with open(ruta, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()

    return {"archivo": nombre_archivo, "contenido": contenido}


@aplicacion.post("/detect")
async def detectar_circulos(
    image:  UploadFile = File(...),
    params: str        = Form(default="{}"),
):
    try:
        parametros = json.loads(params)
    except Exception:
        raise HTTPException(status_code=400, detail="El campo 'params' debe ser un JSON válido.")

    datos_imagen = await image.read()
    try:
        imagen_original = cargar_imagen_desde_bytes(datos_imagen)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    mapa_bordes = preprocesar(imagen_original)
    puntos_borde = obtener_puntos_borde(mapa_bordes)

    if len(puntos_borde) < 3:
        raise HTTPException(
            status_code=422,
            detail="La imagen no tiene suficientes bordes detectados. Prueba con otra imagen."
        )

    detector = DetectorCirculosGA(
        tamanio_poblacion = int(parametros.get("population_size", TAMANIO_POBLACION)),
        prob_cruce        = float(parametros.get("crossover_prob", PROB_CRUCE)),
        prob_mutacion     = float(parametros.get("mutation_prob",  PROB_MUTACION)),
        num_elite         = int(parametros.get("elite_count",      NUM_ELITE)),
        max_generaciones  = int(parametros.get("max_generations",  MAX_GENERACIONES)),
    )

    delta = float(parametros.get("delta", DELTA_TOLERANCIA))

    resultado_ga = detector.detectar(
        puntos_borde, imagen_original.shape,
        delta=delta,
    )

    circulos_detectados = resultado_ga["circulos"]
    mejor_aptitud       = resultado_ga["mejor_aptitud"]

    # MEJORA: error promedio solo sobre inliers del círculo principal
    error_promedio = 0.0
    if circulos_detectados:
        circulo_principal = circulos_detectados[0]
        distancias_al_centro = np.sqrt(
            (puntos_borde[:, 0] - circulo_principal["x"]) ** 2 +
            (puntos_borde[:, 1] - circulo_principal["y"]) ** 2
        )
        desviaciones = np.abs(distancias_al_centro - circulo_principal["r"])
        inliers = desviaciones <= delta
        if np.sum(inliers) > 0:
            error_promedio = float(np.mean(desviaciones[inliers]))

    imagen_anotada = anotar_imagen(imagen_original, circulos_detectados)
    imagen_en_base64 = imagen_a_base64(imagen_anotada)

    return JSONResponse({
        "circles":             circulos_detectados,
        "count":               len(circulos_detectados),
        "avg_error":           round(error_promedio, 4),
        "fitness":             mejor_aptitud,
        "annotated_image_b64": imagen_en_base64,
    })
