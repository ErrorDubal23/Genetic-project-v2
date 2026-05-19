"""
Aquí está el algoritmo genético que busca círculos en la imagen.

La idea general es la misma que la evolución biológica: tenemos una población
de círculos candidatos, los evaluamos, cruzamos los mejores entre sí y mutamos
algunos para explorar nuevas posibilidades. Después de varias generaciones,
el mejor círculo que sobrevivió es nuestra respuesta.

Basado en: Ayala-Ramírez et al. (2006), Pattern Recognition Letters 27, pp. 652-657.

MEJORAS CLAVE:
- El código original tenía un bug donde evaluar_aptitud se llamaba DOS VECES
  consecutivas para cada individuo (código duplicado en fitness.py).
- Cuando un círculo fallaba validación, el original hacía 'continue' en el
  bucle while, lo que intentaba de nuevo con los MISMOS puntos disponibles,
  encontrando el MISMO círculo inválido en bucle infinito (o hasta agotar
  el límite de 20 círculos). Ahora se eliminan los bordes del círculo inválido
  para forzar al GA a buscar en otra zona.
- El duplicado del círculo hacía 'break' del while entero, deteniendo la
  detección de círculos adicionales que podían existir. Ahora se ignora
  el duplicado y se continúa buscando.
- Canonicalización de cromosomas: ordena índices i≤j≤k. El original trataba
  (3,5,8), (5,3,8), (8,5,3) como 6 individuos diferentes que codifican el
  MISMO círculo. Esto desperdiciaba evaluaciones y ralentizaba convergencia.
"""

import numpy as np
from fitness import (
    evaluar_aptitud,
    circulo_desde_tres_puntos,
    construir_rejilla_bordes,
    verificar_continuidad_circulo,
    FRACCION_SECTORES_MINIMA,
    MIN_SECTORES_CONSECUTIVOS,
    MAX_ARCOS_SEPARADOS,
)

# ── Parámetros del algoritmo genético (tomados de la Tabla 1 del paper) ───────
TAMANIO_POBLACION  = 70    # Cuántos círculos candidatos manejamos a la vez
PROB_CRUCE         = 0.55  # Probabilidad de que dos candidatos intercambien información
PROB_MUTACION      = 0.10  # Probabilidad de que un candidato cambie aleatoriamente
NUM_ELITE          = 2     # Cuántos de los mejores pasan directo a la siguiente generación
MAX_GENERACIONES   = 500   # Cuántas vueltas da el algoritmo en total
DELTA_TOLERANCIA   = 2.0   # Margen en píxeles para decidir si un punto está "sobre" el borde

# ── Cómo se arma cada candidato ───────────────────────────────────────────────
GENES_POR_INDIVIDUO     = 3  # Cada candidato se define por 3 puntos de borde
NUMERO_REINICIOS        = 3  # MEJORA: antes 5, ahora 3.
                              # 5 reinicios × 100 gen = 500 evaluaciones totales.
                              # Con el bug de evaluación duplicada, eran 1000.
                              # Ahora 3 × 167 gen = 501 evaluaciones, sin duplicar.
                              # Menos reinicios = menos varianza aleatoria,
                              # más generaciones por reinicio = mejor convergencia.

INTENTOS_INICIALIZACION = 8  # Intentos para encontrar 3 puntos bien separados al inicio
DISTANCIA_MINIMA_PUNTOS = 20 # Los 3 puntos semilla deben estar al menos a esta
                              # distancia entre sí (en píxeles)

# ── Qué tan bueno debe ser un círculo para aceptarlo ─────────────────────────
APTITUD_MINIMA_PRIMER   = 0.35  # MEJORA: antes 0.40, ahora 0.35.
                                 # 0.40 era demasiado estricto para círculos con
                                 # bordes irregulares, ruido moderado o poco
                                 # contraste. 0.35 acepta círculos reales más
                                 # débiles sin bajar tanto la barrera contra
                                 # falsos positivos (la validación angular los filtra).

APTITUD_MINIMA_RESTO    = 0.45  # MEJORA: antes 0.50, ahora 0.45.
                                 # Similar al anterior. Después de eliminar el
                                 # primer círculo, los puntos restantes son
                                 # más ruidosos. 0.45 es más realista.

FRACCION_APTITUD_MINIMA = 0.50  # MEJORA: antes 0.55, ahora 0.50.
                                 # Círculos adicionales deben ser al menos 50%
                                 # tan buenos como el primero. 55% descartaba
                                 # círculos secundarios legítimos que eran
                                 # simplemente más pequeños o parcialmente ocluidos.

# ── Control del proceso de búsqueda ──────────────────────────────────────────
MARGEN_SUPRESION         = 2.5  # MEJORA: antes 3, ahora 2.5.
                                 # 3×delta = 6px de anillo borrado. Si dos
                                 # círculos están a 12px de distancia, el borrado
                                 # de uno destruía bordes del otro.
                                 # 2.5×delta = 5px, conserva más bordes
                                 # de círculos cercanos legítimos.

MAX_CIRCULOS_POSIBLES    = 20  # Límite de seguridad: nunca reportamos más de 20 círculos
MINIMOS_PUNTOS_RESTANTES = 30  # MEJORA: antes 50, ahora 30.
                                 # 50 puntos restantes para detectar un círculo
                                 # adicional era excesivo. Un círculo pequeño
                                 # puede estar definido por 30-40 puntos de borde
                                 # si son buenos. 30 permite detectar círculos
                                 # más pequeños sin caer en ruido puro.

UMBRAL_DUPLICADO_CENTRO  = 12  # MEJORA: antes 10, ahora 12.
                                 # Si dos círculos reales están a 10-11px
                                 # (como burbujas o monedas juntas), el umbral
                                 # de 10 los fusionaba. 12 es más conservador.

UMBRAL_DUPLICADO_RADIO   = 12  # MEJORA: antes 15, ahora 12.
                                 # 15px de diferencia en radio podía fusionar
                                 # círculos concéntricos (aros) que son reales.
                                 # 12px distingue mejor círculos anidados.


class DetectorCirculosGA:
    """
    Esta clase hace todo el trabajo de detección.

    El proceso es:
      1. Toma los puntos de borde de la imagen.
      2. Crea una población de círculos candidatos (cada uno definido por 3 puntos).
      3. Los evoluciona durante varias generaciones para encontrar el mejor.
      4. Valida que lo encontrado sea realmente un círculo y no un polígono.
      5. Borra ese círculo de la imagen y repite para encontrar el siguiente.
    """

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

    # ── Crear la población inicial ────────────────────────────────────────────

    def _separacion_minima(self, indices: np.ndarray,
                           puntos_borde: np.ndarray) -> float:
        # Calcula qué tan separados están los 3 puntos entre sí
        p1, p2, p3 = puntos_borde[indices]
        d12 = float(np.sqrt(np.sum((p1 - p2) ** 2)))
        d13 = float(np.sqrt(np.sum((p1 - p3) ** 2)))
        d23 = float(np.sqrt(np.sum((p2 - p3) ** 2)))
        return min(d12, d13, d23)

    def _inicializar_poblacion(self, num_puntos: int,
                               puntos_borde: np.ndarray) -> np.ndarray:
        """
        Crea los círculos candidatos iniciales.
        Para cada uno intentamos elegir 3 puntos que estén bien separados,
        porque 3 puntos muy juntos definen un círculo muy pequeño y poco útil.
        """
        poblacion = np.zeros((self.tamanio_poblacion, GENES_POR_INDIVIDUO), dtype=int)

        for i in range(self.tamanio_poblacion):
            mejor     = np.random.choice(num_puntos, GENES_POR_INDIVIDUO, replace=False)
            mejor_sep = self._separacion_minima(mejor, puntos_borde)

            for _ in range(INTENTOS_INICIALIZACION):
                candidato = np.random.choice(num_puntos, GENES_POR_INDIVIDUO, replace=False)
                sep = self._separacion_minima(candidato, puntos_borde)
                if sep > mejor_sep:
                    mejor     = candidato
                    mejor_sep = sep

            # MEJORA: canonicalización — ordenar índices para que (3,5,8) = (5,3,8)
            poblacion[i] = np.sort(mejor)

        return poblacion

    # ── Evaluar toda la población ─────────────────────────────────────────────

    def _evaluar_poblacion(self, poblacion: np.ndarray, puntos_borde: np.ndarray,
                           rejilla_bordes: np.ndarray,
                           forma_imagen: tuple, delta: float) -> np.ndarray:
        # Le pone nota a cada candidato usando la función de aptitud
        aptitudes = [
            evaluar_aptitud(ind, puntos_borde, rejilla_bordes, forma_imagen, delta)
            for ind in poblacion
        ]
        return np.array(aptitudes)

    # ── Selección por ruleta ──────────────────────────────────────────────────

    def _seleccion_ruleta(self, poblacion: np.ndarray,
                          aptitudes: np.ndarray) -> np.ndarray:
        """
        Elige qué candidatos pasan a la siguiente generación.
        Los mejores tienen más probabilidad de ser elegidos, pero los peores
        también tienen alguna oportunidad (como una ruleta donde los mejores
        tienen una porción más grande).
        """
        suma = aptitudes.sum()
        if suma == 0:
            probs = np.ones(self.tamanio_poblacion) / self.tamanio_poblacion
        else:
            probs = aptitudes / suma

        indices = np.random.choice(self.tamanio_poblacion,
                                   size=self.tamanio_poblacion,
                                   replace=True, p=probs)
        return poblacion[indices]

    # ── Cruce entre candidatos ────────────────────────────────────────────────

    def _cruce_un_punto(self, poblacion: np.ndarray) -> np.ndarray:
        """
        Combina pares de candidatos para crear nuevos.
        Tomamos dos candidatos, elegimos un punto de corte al azar y
        intercambiamos la parte final de sus genes, como mezclar dos recetas.
        """
        descendencia = poblacion.copy()
        for i in range(0, self.tamanio_poblacion - 1, 2):
            if np.random.rand() < self.prob_cruce:
                corte = np.random.randint(1, GENES_POR_INDIVIDUO)
                tmp                         = poblacion[i,     corte:].copy()
                descendencia[i,     corte:] = poblacion[i + 1, corte:]
                descendencia[i + 1, corte:] = tmp
        # MEJORA: canonicalizar después del cruce para mantener i≤j≤k
        for j in range(len(descendencia)):
            descendencia[j] = np.sort(descendencia[j])
        return descendencia

    # ── Mutación ──────────────────────────────────────────────────────────────

    def _mutar(self, poblacion: np.ndarray, num_puntos: int) -> np.ndarray:
        """
        Cambia aleatoriamente uno de los 3 puntos de algunos candidatos.
        Esto evita que todos converjan a la misma solución y permite explorar
        partes de la imagen que de otra forma no se revisarían.
        """
        for i in range(self.tamanio_poblacion):
            if np.random.rand() < self.prob_mutacion:
                gen   = np.random.randint(GENES_POR_INDIVIDUO)
                otros = set(poblacion[i]) - {int(poblacion[i, gen])}
                nuevo = np.random.randint(num_puntos)
                intentos = 0
                while nuevo in otros and intentos < 15:
                    nuevo    = np.random.randint(num_puntos)
                    intentos += 1
                poblacion[i, gen] = nuevo
                # MEJORA: re-ordenar después de mutación para mantener canonicalización
                poblacion[i] = np.sort(poblacion[i])
        return poblacion

    # ── Una corrida completa del GA ───────────────────────────────────────────

    def _ejecutar_una_vez(self, puntos_borde: np.ndarray,
                          rejilla_bordes: np.ndarray,
                          forma_imagen: tuple, delta: float,
                          num_generaciones: int):
        # Arranca con una población inicial y la hace evolucionar
        num_puntos = len(puntos_borde)
        poblacion  = self._inicializar_poblacion(num_puntos, puntos_borde)
        mejor_ind  = None
        mejor_apt  = -1.0

        for _ in range(num_generaciones):
            aptitudes = self._evaluar_poblacion(
                poblacion, puntos_borde, rejilla_bordes, forma_imagen, delta
            )
            orden     = np.argsort(aptitudes)
            elite     = poblacion[orden[-self.num_elite:]].copy()
            apt_elite = aptitudes[orden[-self.num_elite:]]

            # Guardamos el mejor que hayamos visto en toda la corrida
            if float(apt_elite[-1]) > mejor_apt:
                mejor_apt = float(apt_elite[-1])
                mejor_ind = elite[-1].copy()

            seleccionados = self._seleccion_ruleta(poblacion, aptitudes)
            descendencia  = self._cruce_un_punto(seleccionados)
            descendencia  = self._mutar(descendencia, num_puntos)
            descendencia[:self.num_elite] = elite  # Los élite siempre sobreviven
            poblacion = descendencia

        return mejor_ind, mejor_apt

    # ── Buscar el mejor círculo posible ───────────────────────────────────────

    def _buscar_mejor_circulo(self, puntos_borde: np.ndarray,
                               forma_imagen: tuple, delta: float):
        """
        Corre el GA varias veces desde cero y se queda con el mejor resultado.
        Hacemos esto porque el GA es aleatorio y a veces puede quedarse
        atascado en una solución mediocre.
        """
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

    # ── Confirmar que el círculo es real ──────────────────────────────────────

    def _validar_circulo(self, circulo: dict, rejilla_bordes: np.ndarray,
                          forma_imagen: tuple) -> bool:
        """
        Antes de aceptar un círculo, verificamos que sus bordes formen
        un arco continuo y no sean puntos dispersos de un polígono.

        Tres condiciones que debe cumplir:
          1. Al menos el 25% de la circunferencia tiene borde (cobertura mínima).
          2. Hay un tramo continuo de al menos 75° con borde seguido.
          3. Los bordes están agrupados en máximo 3 pedazos separados.

        Un círculo real, aunque esté parcialmente tapado, cumple las tres.
        Un polígono falla al menos una: sus lados solo cruzan el círculo
        en puntos aislados, sin formar arcos continuos.
        """
        fraccion, max_cons, num_arcos = verificar_continuidad_circulo(
            circulo["x"], circulo["y"], circulo["r"],
            rejilla_bordes, forma_imagen
        )
        return (fraccion    >= FRACCION_SECTORES_MINIMA
                and max_cons >= MIN_SECTORES_CONSECUTIVOS
                and num_arcos <= MAX_ARCOS_SEPARADOS)

    # ── Revisar si ya detectamos ese círculo antes ────────────────────────────

    def _es_duplicado(self, circulo_nuevo: dict,
                       circulos_encontrados: list) -> bool:
        cx = circulo_nuevo["x"]
        cy = circulo_nuevo["y"]
        r  = circulo_nuevo["r"]
        for c in circulos_encontrados:
            dist   = ((cx - c["x"]) ** 2 + (cy - c["y"]) ** 2) ** 0.5
            dradio = abs(r - c["r"])
            if dist < UMBRAL_DUPLICADO_CENTRO and dradio < UMBRAL_DUPLICADO_RADIO:
                return True
        return False

    # ── Detección completa: buscar todos los círculos de la imagen ────────────

    def detectar(self, puntos_borde: np.ndarray, forma_imagen: tuple,
                 delta: float = DELTA_TOLERANCIA) -> dict:
        """
        Detecta todos los círculos de la imagen uno por uno.

        El proceso es:
          1. Buscamos el mejor círculo en los bordes disponibles.
          2. Si no es suficientemente bueno, paramos.
          3. Si no pasa la validación, ELIMINAMOS SUS BORDES y seguimos buscando.
             MEJORA CRÍTICA: el original hacía 'continue' sin eliminar bordes,
             encontrando el mismo círculo inválido en bucle infinito.
          4. Si pasa todo, lo registramos y borramos sus bordes de la imagen.
          5. Repetimos con los bordes restantes.

        Borramos los bordes del círculo encontrado (no todo el disco, sino solo
        el anillo del borde) para no afectar los bordes de otros círculos
        cercanos que todavía no hemos detectado.
        """
        if len(puntos_borde) < GENES_POR_INDIVIDUO:
            return {"circulos": [], "mejor_aptitud": 0.0}

        circulos_encontrados  = []
        aptitudes_encontradas = []
        puntos_disponibles    = puntos_borde.copy()
        aptitud_referencia    = None

        while len(circulos_encontrados) < MAX_CIRCULOS_POSIBLES:
            if len(puntos_disponibles) < MINIMOS_PUNTOS_RESTANTES:
                break

            circulo, aptitud, rejilla_actual = self._buscar_mejor_circulo(
                puntos_disponibles, forma_imagen, delta
            )

            if circulo is None:
                break

            # Si el mejor candidato posible es muy débil, ya no hay círculos
            if aptitud < APTITUD_MINIMA_PRIMER:
                break

            # Para los círculos 2, 3, 4… exigimos que sean casi tan buenos
            # como el primero o que superen el umbral absoluto mínimo
            if aptitud_referencia is not None:
                umbral = max(APTITUD_MINIMA_RESTO,
                             aptitud_referencia * FRACCION_APTITUD_MINIMA)
                if aptitud < umbral:
                    break

            # Si los bordes no forman un arco continuo, no es un círculo real
            if not self._validar_circulo(circulo, rejilla_actual, forma_imagen):
                # MEJORA CRÍTICA: eliminar bordes del círculo inválido para
                # forzar al GA a buscar en otra zona. El original hacía
                # 'continue' sin eliminar, encontrando el mismo círculo
                # inválido repetidamente.
                cx, cy, r = circulo["x"], circulo["y"], circulo["r"]
                dist = np.sqrt(
                    (puntos_disponibles[:, 0] - cx) ** 2 +
                    (puntos_disponibles[:, 1] - cy) ** 2
                )
                # Borramos un anillo más ancho para asegurar que no volvemos
                # a caer en el mismo mínimo local
                fuera_anillo = np.abs(dist - r) > delta * (MARGEN_SUPRESION + 1)
                puntos_disponibles = puntos_disponibles[fuera_anillo]
                continue

            # Si ya encontramos uno casi igual, lo ignoramos y seguimos
            # MEJORA: el original hacía 'break' aquí, deteniendo toda la
            # detección. Ahora seguimos buscando círculos adicionales.
            if self._es_duplicado(circulo, circulos_encontrados):
                cx, cy, r = circulo["x"], circulo["y"], circulo["r"]
                dist = np.sqrt(
                    (puntos_disponibles[:, 0] - cx) ** 2 +
                    (puntos_disponibles[:, 1] - cy) ** 2
                )
                fuera_anillo = np.abs(dist - r) > delta * MARGEN_SUPRESION
                puntos_disponibles = puntos_disponibles[fuera_anillo]
                continue

            if aptitud_referencia is None:
                aptitud_referencia = aptitud

            circulos_encontrados.append(circulo)
            aptitudes_encontradas.append(aptitud)

            # Borramos los bordes del círculo recién encontrado
            # Solo borramos el anillo del borde, no el disco completo,
            # para no afectar círculos cercanos que todavía no detectamos
            cx, cy, r = circulo["x"], circulo["y"], circulo["r"]
            dist = np.sqrt(
                (puntos_disponibles[:, 0] - cx) ** 2 +
                (puntos_disponibles[:, 1] - cy) ** 2
            )
            fuera_anillo       = np.abs(dist - r) > delta * MARGEN_SUPRESION
            puntos_disponibles = puntos_disponibles[fuera_anillo]

        mejor_aptitud_global = aptitudes_encontradas[0] if aptitudes_encontradas else 0.0

        return {
            "circulos":      circulos_encontrados,
            "mejor_aptitud": mejor_aptitud_global,
        }
