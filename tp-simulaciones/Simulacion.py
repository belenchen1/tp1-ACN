
#Simulacion.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Type, Dict, Any, Optional

import numpy as np

from Constants import DAY_END, DAY_START
from Helpers import velocidad_por_distancia, BASELINE_TIME_MIN
from TraficoAEP import TraficoAviones

#Clase Metricas para luego comparar resultados para distintos niveles de lambda
@dataclass
class Metricas:
    lam: float
    arrivals: int
    landed: int
    diverted: int
    minutos_aviones: int
    minutos_aviones_cong: int
    minutos_delay_promedio: float
    delays: List[float]


#------------------------------------------------------------------------------------------

def simular_jornada(ctrl_seed: int, 
                    lam_per_min: float,
                    Controller: Type = TraficoAviones,
                    controller_kwargs: Optional[Dict[str, Any]] = None,
                    t0: Optional[int] = None,
                    t1: Optional[int] = None,
                    ) -> Metricas:
    #Simulacion de una jornada de 18 horas para un λ

    t0 = DAY_START if t0 is None else t0
    t1 = DAY_END if t1 is None else t1
    controller_kwargs = controller_kwargs or {}

    ctrl = Controller(seed = ctrl_seed, **controller_kwargs)


    arribos_set = set(ctrl.bernoulli_aparicion(lam_per_min, t0=t0, t1=t1))
    arribos = len(arribos_set)

    minutos_aviones = 0
    minutos_aviones_cong = 0
    landed_prev: Set[int] = set()
    delays: List[float] = []

    def landed_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "landed"}
    
    def diverted_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "diverted"}


    for t in range(t0, t1):
        ctrl.step(t, aparicion=(t in arribos_set))

        #congestion: minutos_aviones con velociad < vmax en approach
        for aid in ctrl.activos:
            av = ctrl.planes[aid]
            vmin, vmax = velocidad_por_distancia(av.distancia_nm)
            minutos_aviones += 1
            if av.velocidad_kts < vmax - 1e-9:
                minutos_aviones_cong += 1

        #nuevos aterrizados: atraso = t_landig - (t_aparicion + T_base)
        now_landed = landed_set()
        new_landed = now_landed - landed_prev
        for aid in new_landed:
            av = ctrl.planes[aid]
            raw = getattr(av, "aterrizaje_min_cont", None)      #Read de t_land tolerante a None -> derivaba en un problema en experimentacion
            if raw is None:                                 
                raw = getattr(av, "aterrizaje_min_continuo", None)

            try:
                t_land = float(raw) if raw is not None else float(t)
            except (TypeError, ValueError):
                t_land = float(t)
            
            llegada_esperada = av.aparicion_min + BASELINE_TIME_MIN
            delays.append(t_land - llegada_esperada)
        landed_prev = now_landed
    
    diverted = len(diverted_set())
    landed = len(landed_prev)
    minutos_delay_promedio = float(np.mean(delays)) if delays else float('nan')

    return Metricas(
        lam = lam_per_min,
        arrivals = arribos,
        landed = landed,
        diverted = diverted,
        minutos_aviones = minutos_aviones,
        minutos_aviones_cong = minutos_aviones_cong,
        minutos_delay_promedio = minutos_delay_promedio,
        delays = delays
    )



def simular_jornada_base(ctrl_seed: int, lam_per_min: float, **kwargs) -> Metricas:
    """Escenario base (TraficoAviones)."""
    return simular_jornada(ctrl_seed, lam_per_min, Controller=TraficoAviones, controller_kwargs=kwargs or None)

def simular_jornada_viento(ctrl_seed: int, lam_per_min: float, **kwargs) -> Metricas:
    """Escenario con viento/go-around (TraficoAEPViento)."""
    from TraficoAEPViento import TraficoAEPViento  # import local para evitar ciclos/errores si no existe aún
    return simular_jornada(ctrl_seed, lam_per_min, Controller=TraficoAEPViento, controller_kwargs=kwargs or None)

def simular_jornada_cierre(ctrl_seed: int, lam_per_min: float, start_min: int, dur_min: int, **kwargs) -> Metricas:
    """Escenario con cierre de AEP (TraficoAEPCerrado)."""
    from TraficoAEPCierre import TraficoAEPCerrado, AEPCerrado
    cierre = AEPCerrado(start_min=start_min, dur_min=dur_min)
    kw = dict(kwargs or {})
    kw["cierre"] = cierre
    return simular_jornada(ctrl_seed, lam_per_min, Controller=TraficoAEPCerrado, controller_kwargs=kw)

def simular_jornada_politica(ctrl_seed: int, lam_per_min: float, **kwargs) -> Metricas:
    """Escenario con Politica de Metering (TraficoAvionesPolitica)"""
    from TraficoAEPPolitica import TraficoAvionesPolitica
    return simular_jornada(
        ctrl_seed, lam_per_min,
        Controller=TraficoAvionesPolitica,
        controller_kwargs=kwargs or None
    )

def simular_jornada_politica2a(ctrl_seed: int, lam_per_min: float, **kwargs) -> Metricas:
    """Escenario con Politica de Prioridad de Reingreso (TraficoAvionesPolitica2a)"""
    from TraficoAEPPolitica import TraficoAvionesPolitica2a
    return simular_jornada(
        ctrl_seed, lam_per_min,
        Controller=TraficoAvionesPolitica2a,
        controller_kwargs=kwargs or None
    )

def simular_jornada_politica2b(ctrl_seed: int, lam_per_min: float, **kwargs) -> Metricas:
    """Escenario con Politica de Prioridad de Reingreso (TraficoAvionesPolitica2b)"""
    from TraficoAEPPolitica import TraficoAvionesPolitica2b
    return simular_jornada(
        ctrl_seed, lam_per_min,
        Controller=TraficoAvionesPolitica2b,
        controller_kwargs=kwargs or None
    )