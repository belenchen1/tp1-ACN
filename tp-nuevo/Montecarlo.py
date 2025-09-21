
# Montecarlo.py

from __future__ import annotations
from typing import List, Dict, Any, Optional, Type
import random, math
import numpy as np
import pandas as pd

from Simulacion import simular_jornada, Metricas
from TraficoAEP import TraficoAviones  # default (escenario base)

__all__ = [
    "ic95_media",
    "montecarlo_dias",
    "montecarlo_base",
    "montecarlo_viento",
    "montecarlo_cierre",
]

# --------------------------- helpers ---------------------------------

def _nan_filtered(xs: List[float]) -> List[float]:
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]

def ic95_media(xs: List[float]) -> tuple[float, float]:
    """
    Intervalo de confianza 95% de la media con z=1.96.
    Devuelve (low, high). Si no alcanza n>=2 -> (nan, nan).
    """
    xs = _nan_filtered(xs)
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (xs[0], xs[0])
    mu = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1))
    se = sd / math.sqrt(n)
    return (mu - 1.96 * se, mu + 1.96 * se)

# --------------------------- MONTECARLO ------------------------------------

def montecarlo_dias(
    lams: List[float],
    dias: int = 100,
    seed: int = 25,
    Controller: Type = TraficoAviones,
    controller_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Corre 'dias' jornadas por cada λ en 'lams' usando el Controller indicado.
    Devuelve un DataFrame con medias e IC95 de: congestión, delay y desvío.

    Compatibilidad: si llamás sin Controller/kwargs usa TraficoAviones (base).
    """
    rng = random.Random(seed)
    controller_kwargs = controller_kwargs or {}

    rows: List[Dict[str, Any]] = []

    for lam in lams:
        daily_cong: List[float] = []
        daily_delay: List[float] = []
        daily_divr: List[float] = []
        daily_arrivals: List[float] = []
        daily_divert_counts: List[float] = []

        for _ in range(dias):
            dm: Metricas = simular_jornada(
                ctrl_seed=rng.randrange(10**9),
                lam_per_min=lam,
                Controller=Controller,
                controller_kwargs=controller_kwargs,
            )

            # tasas diarias
            cong_rate = (dm.minutos_aviones_cong / dm.minutos_aviones) if dm.minutos_aviones else float("nan")
            div_rate  = (dm.diverted / dm.arrivals) if dm.arrivals else float("nan")

            daily_cong.append(cong_rate)
            daily_delay.append(dm.minutos_delay_promedio)
            daily_divr.append(div_rate)
            daily_arrivals.append(dm.arrivals)
            daily_divert_counts.append(dm.diverted)

        # medias (ignorando NaN) + IC95
        cong_mean = float(np.nanmean(daily_cong)) if len(daily_cong) else float("nan")
        delay_mean = float(np.nanmean(daily_delay)) if len(daily_delay) else float("nan")
        divert_mean = float(np.nanmean(daily_divr)) if len(daily_divr) else float("nan")

        cong_ci  = ic95_media(daily_cong)
        delay_ci = ic95_media(daily_delay)
        divert_ci = ic95_media(daily_divr)

        rows.append({
            "lam": lam,

            "congestion_rate_mean": cong_mean,
            "congestion_rate_ci_low": cong_ci[0],
            "congestion_rate_ci_high": cong_ci[1],

            "avg_delay_min_mean": delay_mean,
            "avg_delay_min_ci_low": delay_ci[0],
            "avg_delay_min_ci_high": delay_ci[1],

            "divert_rate_mean": divert_mean,
            "divert_rate_ci_low": divert_ci[0],
            "divert_rate_ci_high": divert_ci[1],

            "avg_arrivals_per_day": float(np.nanmean(daily_arrivals)) if len(daily_arrivals) else float("nan"),
            "avg_diverted_per_day": float(np.nanmean(daily_divert_counts)) if len(daily_divert_counts) else float("nan"),
            "dias": dias,
        })

    return pd.DataFrame(rows)

# --------------------------- wrappers cómodos ------------------------

def montecarlo_base(lams: List[float], dias: int = 100, seed: int = 25, **kwargs) -> pd.DataFrame:
    """Montecarlo con el escenario base."""
    return montecarlo_dias(lams, dias=dias, seed=seed, Controller=TraficoAviones, controller_kwargs=kwargs or None)

def montecarlo_viento(lams: List[float], dias: int = 100, seed: int = 25, **kwargs) -> pd.DataFrame:
    """Montecarlo con viento/go-around (TraficoAEPViento)."""
    from TraficoAEPViento import TraficoAEPViento  # import local para evitar problemas si aún no existe el .py
    return montecarlo_dias(lams, dias=dias, seed=seed, Controller=TraficoAEPViento, controller_kwargs=kwargs or None)

def montecarlo_cierre(lams: List[float], dias: int = 100, seed: int = 25, start_min: int = 180, dur_min: int = 30, **kwargs) -> pd.DataFrame:
    """Montecarlo con cierre de AEP (TraficoAEPCerrado)."""
    from TraficoAEPCierre import TraficoAEPCerrado, AEPCerrado
    kw = dict(kwargs or {})
    kw["closure"] = AEPCerrado(start_min=start_min, dur_min=dur_min)
    return montecarlo_dias(lams, dias=dias, seed=seed, Controller=TraficoAEPCerrado, controller_kwargs=kw)

def montecarlo_politica(lams: List[float], dias: int = 100, seed:int = 25, **kwargs) -> pd.DataFrame:
    """Montecarlo con Politica de Metering (TraficoAvionesPolitica)"""
    from TraficoAEPPolitica import TraficoAvionesPolitica
    return montecarlo_dias(
        lams, dias = dias, seed=seed,
        Controller=TraficoAvionesPolitica, controller_kwargs=kwargs or None
    )

def montecarlo_politica2a(lams: List[float], dias: int = 100, seed:int = 25, **kwargs) -> pd.DataFrame:
    """Montecarlo con Politica de Prioridad de Reingreso por Riesgo de Desvio (TraficoAvionesPolitica2a)"""
    from TraficoAEPPolitica import TraficoAvionesPolitica2a
    return montecarlo_dias(
        lams, dias = dias, seed=seed,
        Controller=TraficoAvionesPolitica2a, controller_kwargs=kwargs or None
    )

def montecarlo_politica2b(lams: List[float], dias: int = 100, seed:int = 25, **kwargs) -> pd.DataFrame:
    """Montecarlo con Politica de Prioridad de Reingreso FIFO por Tiempo en Turnaround (TraficoAvionesPolitica2b)"""
    from TraficoAEPPolitica import TraficoAvionesPolitica2b
    return montecarlo_dias(
        lams, dias = dias, seed=seed,
        Controller=TraficoAvionesPolitica2b, controller_kwargs=kwargs or None
    )