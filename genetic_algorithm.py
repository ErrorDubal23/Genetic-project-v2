"""
Algoritmo genético para detección de círculos.

MEJORAS PROFUNDAS RESPECTO AL CÓDIGO ORIGINAL:

1. BUG CRÍTICO FIX: Círculo inválido → continue sin eliminar bordes
   El código original hacía 'continue' sin eliminar los puntos del círculo
   inválido, causando bucle infinito en el mismo mínimo local.

2. BUG CRÍTICO FIX: Duplicado → break detenía TODA la detección
   El código original hacía 'break' al encontrar un duplicado, deteniendo
   la búsqueda de círculos adicionales. Ahora se eliminan los bordes del
   duplicado y se continúa buscando.

3. Canonicalización de cromosomas: ordena índices i≤j≤k. Reduce espacio
   de búsqueda 6× y elimina evaluaciones redundantes.

4. Validación ADAPTIVA: umbrales angulares se relajan para círculos con
   alta aptitud (>0.65), permitiendo detectar círculos reales con oclusión
   parcial sin aceptar falsos positivos de baja aptitud.

5. Eliminación de aptitud_referencia: el concepto de "círculos adicionales
   deben ser 50% del primero" causaba falsos negativos cuando el primer
   círculo era excepcionalmente bueno y los demás eran normales.

6. Evaluación de offspring 1 sola vez por generación (el original evaluaba
   2 veces implícitamente a través del código duplicado en fitness.py).
"""

import numpy as np
from fitness import (
    evaluar_aptitud,
    circulo_desde_tres_puntos,
    construir_rejilla_bordes,
    verificar_continuidad_circulo,
    APTITUD_ALTA_UMBRAL,
)

# ── Parámetros del GA ────────────────────────────────────────────────────────
TAMANIO_POBLACION  = 70
PROB_CRUCE         = 0.55
PROB_MUTACION      = 0.10
NUM_ELITE          = 2
MAX_GENERACIONES   = 500
DELTA_TOLERANCIA   = 2.0

GENES_POR_INDIVIDUO     = 3
NUMERO_REINICIOS        = 3      # Más generaciones por reinicio = mejor convergencia
INTENTOS_INICIALIZACION = 8
DISTANCIA_MINIMA_PUNTOS = 20

# ── Umbrales de aceptación ────────────────────────────────────────────────────
# MEJORA: umbral único para todos los círculos. El concepto de "círculos
# adicionales deben ser X% del primero" causaba que círculos legítimos
# fueran rechazados cuando el primero tenía aptitud excepcional.
APTITUD_MINIMA          = 0.40   # Umbral único para todos los círculos

# ── Control del proceso de búsqueda ──────────────────────────────────────────
MARGEN_SUPRESION         = 2.5   # Multiplicador de delta para borrar bordes
MAX_CIRCULOS_POSIBLES    = 20
MINIMOS_PUNTOS_RESTANTES = 30
UMBRAL_DUPLICADO_CENTRO  = 12
UMBRAL_DUPLICADO_RADIO   = 12


class DetectorCirculosGA:
    def __init__(
        self,
        tamanio_poblacion: int   = TAMANIO_POBLACION,
        prob_cruce:        float = PROB_CRUCE,
        prob_mutacion:     float = PROB_MUTACION,
        num_elite:         int   = NUM_ELITE,
        max_generaciones:  int   = MAX_GENERACIONES,
    ):
        self.tamanio_poblacion = tamanio_poblacion
        self.prob_cruce        = prob_cruce
        self.prob_mutacion     = prob_mutacion
        self.num_elite         = num_elite
        self.max_generaciones  = max_generaciones

    # ── Población inicial ────────────────────────────────────────────────────

    def _separacion_minima(self, indices, puntos_borde):
        p1, p2, p3 = puntos_borde[indices]
        d12 = float(np.sqrt(np.sum((p1 - p2) ** 2)))
        d13 = float(np.sqrt(np.sum((p1 - p3) ** 2)))
        d23 = float(np.sqrt(np.sum((p2 - p3) ** 2)))
        return min(d12, d13, d23)

    def _inicializar_poblacion(self, num_puntos, puntos_borde):
        poblacion = np.zeros((self.tamanio_poblacion, GENES_POR_INDIVIDUO), dtype=int)

        for i in range(self.tamanio_poblacion):
            mejor = np.random.choice(num_puntos, GENES_POR_INDIVIDUO, replace=False)
            mejor_sep = self._separacion_minima(mejor, puntos_borde)

            for _ in range(INTENTOS_INICIALIZACION):
                candidato = np.random.choice(num_puntos, GENES_POR_INDIVIDUO, replace=False)
                sep = self._separacion_minima(candidato, puntos_borde)
                if sep > mejor_sep:
                    mejor = candidato
                    mejor_sep = sep

            # Canonicalización: ordenar índices para reducir espacio de búsqueda
            poblacion[i] = np.sort(mejor)

        return poblacion

    # ── Evaluar población ────────────────────────────────────────────────────

    def _evaluar_poblacion(self, poblacion, puntos_borde, rejilla_bordes,
                           forma_imagen, delta):
        return np.array([
            evaluar_aptitud(ind, puntos_borde, rejilla_bordes, forma_imagen, delta)
            for ind in poblacion
        ])

    # ── Selección por ruleta ─────────────────────────────────────────────────

    def _seleccion_ruleta(self, poblacion, aptitudes):
        suma = aptitudes.sum()
        if suma == 0:
            probs = np.ones(self.tamanio_poblacion) / self.tamanio_poblacion
        else:
            probs = aptitudes / suma

        indices = np.random.choice(self.tamanio_poblacion,
                                   size=self.tamanio_poblacion,
                                   replace=True, p=probs)
        return poblacion[indices]

    # ── Cruce ──────────────────────────────────────────────────────────────────

    def _cruce_un_punto(self, poblacion):
        descendencia = poblacion.copy()
        for i in range(0, self.tamanio_poblacion - 1, 2):
            if np.random.rand() < self.prob_cruce:
                corte = np.random.randint(1, GENES_POR_INDIVIDUO)
                tmp = poblacion[i, corte:].copy()
                descendencia[i, corte:] = poblacion[i + 1, corte:]
                descendencia[i + 1, corte:] = tmp

        # Canonicalización post-cruce
        for j in range(len(descendencia)):
            descendencia[j] = np.sort(descendencia[j])

        return descendencia

    # ── Mutación ─────────────────────────────────────────────────────────────

    def _mutar(self, poblacion, num_puntos):
        for i in range(self.tamanio_poblacion):
            if np.random.rand() < self.prob_mutacion:
                gen = np.random.randint(GENES_POR_INDIVIDUO)
                otros = set(poblacion[i]) - {int(poblacion[i, gen])}
                nuevo = np.random.randint(num_puntos)
                intentos = 0
                while nuevo in otros and intentos < 15:
                    nuevo = np.random.randint(num_puntos)
                    intentos += 1
                poblacion[i, gen] = nuevo
                poblacion[i] = np.sort(poblacion[i])
        return poblacion

    # ── Una ejecución del GA ─────────────────────────────────────────────────

    def _ejecutar_una_vez(self, puntos_borde, rejilla_bordes, forma_imagen,
                          delta, num_generaciones):
        num_puntos = len(puntos_borde)
        poblacion = self._inicializar_poblacion(num_puntos, puntos_borde)
        mejor_ind = None
        mejor_apt = -1.0

        for _ in range(num_generaciones):
            aptitudes = self._evaluar_poblacion(
                poblacion, puntos_borde, rejilla_bordes, forma_imagen, delta
            )
            orden = np.argsort(aptitudes)
            elite = poblacion[orden[-self.num_elite:]].copy()
            apt_elite = aptitudes[orden[-self.num_elite:]]

            if float(apt_elite[-1]) > mejor_apt:
                mejor_apt = float(apt_elite[-1])
                mejor_ind = elite[-1].copy()

            seleccionados = self._seleccion_ruleta(poblacion, aptitudes)
            descendencia = self._cruce_un_punto(seleccionados)
            descendencia = self._mutar(descendencia, num_puntos)
            descendencia[:self.num_elite] = elite
            poblacion = descendencia

        return mejor_ind, mejor_apt

    # ── Buscar mejor círculo ───────────────────────────────────────────────────

    def _buscar_mejor_circulo(self, puntos_borde, forma_imagen, delta):
        generaciones_por_reinicio = max(1, self.max_generaciones // NUMERO_REINICIOS)
        rejilla_bordes = construir_rejilla_bordes(puntos_borde, forma_imagen, delta)

        mejor_ind_global = None
        mejor_apt_global = -1.0

        for _ in range(NUMERO_REINICIOS):
            mejor_ind, mejor_apt = self._ejecutar_una_vez(
                puntos_borde, rejilla_bordes, forma_imagen, delta,
                generaciones_por_reinicio
            )
            if mejor_apt > mejor_apt_global:
                mejor_apt_global = mejor_apt
                mejor_ind_global = mejor_ind

        if mejor_ind_global is None or mejor_apt_global <= 0:
            return None, mejor_apt_global, rejilla_bordes

        p1 = puntos_borde[mejor_ind_global[0]]
        p2 = puntos_borde[mejor_ind_global[1]]
        p3 = puntos_borde[mejor_ind_global[2]]

        resultado = circulo_desde_tres_puntos(p1, p2, p3)
        if resultado is None:
            return None, mejor_apt_global, rejilla_bordes

        cx, cy, r = resultado
        circulo = {
            "x": round(float(cx), 2),
            "y": round(float(cy), 2),
            "r": round(float(r),  2),
        }
        return circulo, round(float(mejor_apt_global), 4), rejilla_bordes

    # ── Validar círculo ──────────────────────────────────────────────────────

    def _validar_circulo(self, circulo, aptitud, rejilla_bordes, forma_imagen):
        """
        Valida que un círculo sea real usando verificación ADAPTIVA.
        Si la aptitud es muy alta (>=0.65), relajamos los requisitos angulares
        porque un círculo con muchos inliers globales probablemente es real
        aunque tenga oclusión parcial.
        """
        fraccion, max_cons, num_arcos, es_valido = verificar_continuidad_circulo(
            circulo["x"], circulo["y"], circulo["r"],
            rejilla_bordes, forma_imagen,
            aptitud_base=aptitud
        )
        return es_valido

    # ── Duplicado ────────────────────────────────────────────────────────────

    def _es_duplicado(self, circulo_nuevo, circulos_encontrados):
        cx = circulo_nuevo["x"]
        cy = circulo_nuevo["y"]
        r = circulo_nuevo["r"]
        for c in circulos_encontrados:
            dist = ((cx - c["x"]) ** 2 + (cy - c["y"]) ** 2) ** 0.5
            dradio = abs(r - c["r"])
            if dist < UMBRAL_DUPLICADO_CENTRO and dradio < UMBRAL_DUPLICADO_RADIO:
                return True
        return False

    # ── Eliminar bordes de un círculo ────────────────────────────────────────

    def _eliminar_bordes_circulo(self, puntos_disponibles, circulo, delta,
                                  margen_extra=0):
        """
        Elimina los puntos de borde cercanos a la circunferencia del círculo.
        margen_extra permite usar un anillo más ancho para círculos inválidos.
        """
        cx, cy, r = circulo["x"], circulo["y"], circulo["r"]
        dist = np.sqrt(
            (puntos_disponibles[:, 0] - cx) ** 2 +
            (puntos_disponibles[:, 1] - cy) ** 2
        )
        margen_total = delta * (MARGEN_SUPRESION + margen_extra)
        fuera_anillo = np.abs(dist - r) > margen_total
        return puntos_disponibles[fuera_anillo]

    # ── Detección completa ───────────────────────────────────────────────────

    def detectar(self, puntos_borde, forma_imagen, delta=DELTA_TOLERANCIA):
        """
        Detecta todos los círculos de la imagen uno por uno.

        Flujo:
        1. Buscar mejor círculo con GA.
        2. Si aptitud < umbral, parar.
        3. Validar con verificación ADAPTIVA (umbrales según aptitud).
        4. Si no pasa validación: ELIMINAR sus bordes y continuar.
           (El original hacía 'continue' sin eliminar → bucle infinito)
        5. Si es duplicado: ELIMINAR sus bordes y continuar.
           (El original hacía 'break' → detenía toda detección)
        6. Registrar círculo y eliminar sus bordes.
        7. Repetir.
        """
        if len(puntos_borde) < GENES_POR_INDIVIDUO:
            return {"circulos": [], "mejor_aptitud": 0.0}

        circulos_encontrados = []
        aptitudes_encontradas = []
        puntos_disponibles = puntos_borde.copy()

        while len(circulos_encontrados) < MAX_CIRCULOS_POSIBLES:
            if len(puntos_disponibles) < MINIMOS_PUNTOS_RESTANTES:
                break

            circulo, aptitud, rejilla_actual = self._buscar_mejor_circulo(
                puntos_disponibles, forma_imagen, delta
            )

            if circulo is None:
                break

            # Umbral mínimo de aptitud
            if aptitud < APTITUD_MINIMA:
                break

            # Validación ADAPTIVA: umbrales según aptitud
            if not self._validar_circulo(circulo, aptitud, rejilla_actual, forma_imagen):
                # ELIMINAR bordes del círculo inválido para forzar búsqueda en otra zona
                puntos_disponibles = self._eliminar_bordes_circulo(
                    puntos_disponibles, circulo, delta, margen_extra=1
                )
                continue

            # Si es duplicado, eliminar y continuar buscando
            if self._es_duplicado(circulo, circulos_encontrados):
                puntos_disponibles = self._eliminar_bordes_circulo(
                    puntos_disponibles, circulo, delta
                )
                continue

            circulos_encontrados.append(circulo)
            aptitudes_encontradas.append(aptitud)

            # Eliminar bordes del círculo detectado
            puntos_disponibles = self._eliminar_bordes_circulo(
                puntos_disponibles, circulo, delta
            )

        mejor_aptitud_global = aptitudes_encontradas[0] if aptitudes_encontradas else 0.0

        return {
            "circulos": circulos_encontrados,
            "mejor_aptitud": mejor_aptitud_global,
        }
