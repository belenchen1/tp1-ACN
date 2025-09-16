from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
import random
from main import TraficoAviones, DAY_START, DAY_END


def simular_una_jornada(ctrl_seed: int, lam_per_min: float) -> Tuple[List[int], List[int], int]:
    """
    Corre una jornada de 18 horas (DAY_END - DAY_START) y devuelve:
    - aterrizajes_por_hora: lista de enteros con aterrizajes por hora
    - arribos_por_hora: lista de enteros con 'apariciones' por hora (arrivals del Poisson)
    - desviados: cantidad de aviones que terminaron 'diverted' en el día
    """
    ctrl = TraficoAviones(seed=ctrl_seed) # cambia la seed porque cada día es independiente del otro
    apariciones = set(ctrl.bernoulli_aparicion(lam_per_min, t0=DAY_START, t1=DAY_END)) 

    num_horas = (DAY_END - DAY_START) // 60
    # inicializo dos listas con ceros de longitud num_horas
    aterrizajes_por_hora = [0] * num_horas
    arribos_por_hora = [0] * num_horas

    # contar arribos por hora directamente desde las apariciones (Poisson)
    for t in apariciones:
        h = (t - DAY_START) // 60 # min en el que aparece, convertido a hora (le resto DAY_START por si no es 0 como lo tomamos nosotros, ej 6am)
        if 0 <= h < num_horas: # chequeo que esté en el rango válido
            arribos_por_hora[h] += 1

    prev_landed: Set[int] = set() # aviones que ya habían aterrizado para después no tomarlos en cuenta

    def landed_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "landed"}

    def diverted_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "diverted"}

    for t in range(DAY_START, DAY_END):
        ctrl.step(t, aparicion=(t in apariciones))

        now_landed = landed_set()
        nuevos_landed = now_landed - prev_landed # me quedo con los que aterrizaron justo en este minuto
        if nuevos_landed:
            h = (t - DAY_START) // 60
            if 0 <= h < num_horas:
                aterrizajes_por_hora[h] += len(nuevos_landed)
        prev_landed = now_landed

    total_diverted = len(diverted_set())
    return aterrizajes_por_hora, arribos_por_hora, total_diverted


def ic95_proporcion(p_sombrero: float, n: int) -> Tuple[float, float]:
    '''
    devuelve el intervalo de confianza al 95% para una proporción p_sombrero
    p_sombrero: proporción muestral
    n: tamaño de la muestra
    '''
    if n == 0:
        return (float("nan"), float("nan"))
    # misma fórmula, escrita más explícita
    se = math.sqrt(p_sombrero * (1 - p_sombrero) / n) # standard error
    # 1.96 es el z* para 95% (distribución normal estándar)
    low = max(0.0, p_sombrero - 1.96 * se) 
    high = min(1.0, p_sombrero + 1.96 * se)
    return low, high


def montecarlo_dias(lam_per_min: float = 1.0/60.0, dias: int = 90, seed: int = 12345):
    """
    Corre Monte Carlo por 'dias' o jornadas (18h c/u) y estima:
    - p_landed(X=5): prob. de exactamente 5 aterrizajes en una hora del sistema
    - p_arrivals(Y=5): prob. de exactamente 5 arribos Poisson en una hora
    - IC 95% para ambas
    También reporta desvíos totales.
    """
    rng = random.Random(seed)
    horas_totales = 0
    horas_con_5_landed = 0
    horas_con_5_arrivals = 0
    desviados_totales = 0

    muestra_horas_landed: List[int] = []
    muestra_horas_arrivals: List[int] = []

    for _ in range(dias):
        ctrl_seed = rng.randrange(1_000_000_000) # nueva seed para cada día
        aterr_por_hora, arribos_por_hora, desviados = simular_una_jornada(ctrl_seed, lam_per_min)
        desviados_totales += desviados

        # x_l son los aterrizajes/landings en una hora, x_a los arribos/arrivals en esa misma hora
        for x_l, x_a in zip(aterr_por_hora, arribos_por_hora):
            horas_totales += 1
            muestra_horas_landed.append(x_l)
            muestra_horas_arrivals.append(x_a)
            if x_l == 5:
                horas_con_5_landed += 1 #contador para metricas
            if x_a == 5:
                horas_con_5_arrivals += 1 #idem

    p_landed = horas_con_5_landed / horas_totales if horas_totales else float("nan") 
    p_arrivals = horas_con_5_arrivals / horas_totales if horas_totales else float("nan")
    ic_landed = ic95_proporcion(p_landed, horas_totales)
    ic_arrivals = ic95_proporcion(p_arrivals, horas_totales)

    print(f"=== Monte Carlo {dias} días ===")
    print(f"λ por minuto: {lam_per_min:.6f}  (≈ {lam_per_min*60:.3f} arribos esperados por hora)")
    print(f"Días simulados: {dias}")
    print(f"Horas totales: {horas_totales}")
    print(f"[LAND] Horas con exactamente 5 aterrizajes: {horas_con_5_landed}")
    print(f"[LAND] p(X=5) = {p_landed:.6f}  IC95%: [{ic_landed[0]:.6f}, {ic_landed[1]:.6f}]")
    print(f"[ARR ] Horas con exactamente 5 arribos:     {horas_con_5_arrivals}")
    print(f"[ARR ] p(Y=5) = {p_arrivals:.6f}  IC95%: [{ic_arrivals[0]:.6f}, {ic_arrivals[1]:.6f}]")
    print(f"Desvíos totales en {dias} días: {desviados_totales}")

    return {
        "landings": {
            "p_5": p_landed,
            "ic95": ic_landed,
            "horas_totales": horas_totales,
            "horas_con_5": horas_con_5_landed,
            "muestra_horas": muestra_horas_landed,
        },
        "arrivals": {
            "p_5": p_arrivals,
            "ic95": ic_arrivals,
            "horas_totales": horas_totales,
            "horas_con_5": horas_con_5_arrivals,
            "muestra_horas": muestra_horas_arrivals,
        },
        "desviados_totales": desviados_totales,
    }


if __name__ == "__main__":
    resultados = montecarlo_dias(lam_per_min=1.0/60.0, dias=1000, seed=2030)
    # print(resultados)
