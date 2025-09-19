# -*- coding: utf-8 -*-
from __future__ import annotations
import math, os, random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Simulador base y funciones helpers
from main import TraficoAviones, DAY_START, DAY_END, velocidad_por_distancia, knots_to_nm_per_min


#-------------------------------------------------------------------------
# Ejercicio 6: La idea es que en un momento dado del dia se cierre AEP y queremos observar
# como influye este cierre a las metricas de la simulacion -> suponemos que si cierra, el siguiente
# avion a aterrizar queda "blockeando" la pista de aterrizaje a los aviones restantes de la fila, los cuales
# deben reaccionar dadas las reglas de la simulacion ya establecidas.
#-------------------------------------------------------------------------


@dataclass
class AEPCerrado:
    start_min: int = 180
    dur_min: int = 30

    #funcion para ver si en el instante t AEP esta cerrado o no...
    def is_closed(self, t: int) -> bool:
        return self.start_min <= t < (self.start_min + self.dur_min)
    

class TraficoAEPCerrado(TraficoAviones):
    """
    Cierre AEP por 30 min: el primer avión que llega a 0 nm durante la ventana NO aterriza,
    queda 'blocked' en (0 nm, 0 kts) hasta la reapertura; el resto reacciona con las políticas
    del simulador (baja 20 kts, vmin, turnaround y reingreso con gap>=10 si hace falta).
    """
#Nota: el super sirve para usar metodos de una clase padre -> en este caso TraficoAEPCerrado es hijo de TraficoAviones
    def __init__(self, closure: AEPCerrado, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closure = closure
        self.blocking_id: Optional[int] = None
        self.appear_time: Dict[int, int] = {}
        self.land_time: Dict[int, int] = {}
    
    def aparcion(self, minuto:int):
        av = super().aparicion(minuto)
        self.appear_time[av.id] = minuto
        return av
    

    def mover_paso(self) -> None:
       
        #APPROACH
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * 1.0
            new_dist = max(0.0, av.distancia_nm - avance_nm)
            would_land = (new_dist <= 0) #va a ser true si en el siguiente paso le tocaria aterrizar

            #Si en el siguiente paso tendria que aterrizar y AEP esta cerrado:
                #Si no hay un avion ya blockeando se pone como avion que "blockea"
                #Si ya hay un avion blockeando lo pego al que esta blockeando para que en control reaccione segun las restricciones
            #Si no estaba cerrado lo aterrizo normalmente

            if would_land and self.closure.is_closed(getattr(self, "current_minute", 0)):
                if self.blocking_id is None:
                    self.blocking_id = aid
                    av.distancia_nm = 0.0
                    av.velocidad_kts = 0.0
                    av.estado = "blocked"
                
                else:
                    av.distancia_nm = 0.01
            else:
                av.distancia_nm = new_dist
                if av.distancia_nm <= 0.0:
                    av.distancia_nm = 0.0
                    av.velocidad_kts = 0.0
                    av.estado = "landed"
                    self.mover_a_inactivos(aid)
                    self.land_time[aid] = getattr(self, "current_minute", 0)

        
    
        #TURNAROUND: va a ser igual que en la clase base
        for aid in list(self.turnaround):
            av = self.planes[aid]
            if aid in self.recien_turnaround:
                continue
            retro_nm = knots_to_nm_per_min(self.VELTURNAROUND) * 1.0
            av.distancia_nm += retro_nm
            if av.distancia_nm >= self.MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)


        #REAPERTURA: El avion que estaba bloqueando aterriza y se vuelve a reanudar la simulacion
        if (not self.closure.is_closed(getattr(self, "current_minute", 0))) and (self.blocking_id is not None):
            bid = self.blocking_id
            if bid in self.activos:
                avion_blockeador = self.planes[bid]
                avion_blockeador.estado = "landed"
                self.mover_a_inactivos(bid)
                self.land_time[bid] = getattr(self, "current_minute", 0)
            self.blocking_id = None

        
        self.ordenar_activos()

        
        #Se define el paso pero para el caso con AEP cerrado
        def step(self, minuto: int, aparicion: bool) -> None:
            self.current_minute = minuto
            super().step(minuto, aparicion)


#-------------------------------------------------------------------------
# Nuevamente se define el baseline que calcula el tiempo que tarda un avion desde
# las 100mn hasta AEP si puede ir en todo momento en la vmax que su banda le permite
# sirve para calcular delays luego
#-------------------------------------------------------------------------

    
def baseline_time_from_100nm(step_nm: float = 0.1) -> float:
    d = 100.0
    t = 0.0
    while d > 0:
        vmin, vmax = velocidad_por_distancia(d)
        v_nm_min = knots_to_nm_per_min(vmax)
        ds = min(step_nm, d)
        t += ds / v_nm_min
        d -= ds
    return t

BASELINE_TIME_MIN = baseline_time_from_100nm()



#-------------------------------------------------------------------------
# Una vez que se definio la simulacion con el caso de AEP cerrado vuelvo a definir 
# la clase metricas para sacar resultados valiosos con ella
#-------------------------------------------------------------------------


@dataclass
class Metricas:
    lam: float
    arrivals: int
    landed: int
    diverted: int
    avg_delay_min: float
    cong_rate: float          
    delays: List[float]

def _landed_set(ctrl: TraficoAviones) -> Set[int]:
    return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "landed"}

def _diverted_set(ctrl: TraficoAviones) -> Set[int]:
    return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "diverted"}

#Se simular una jornada (18H) con cierre AEP luego de 180 minutos operativos y mide metricas
def simular_jornada_6(ctrl_seed: int, lam_per_min: float, inicio_cierre: int = 180, duracion_cierre: int = 30) -> Metricas:
    
    ctrl = TraficoAEPCerrado(closure=AEPCerrado(inicio_cierre, duracion_cierre), seed = ctrl_seed)
    arribos_set = set(ctrl.bernoulli_aparicion(lam_per_min, t0=DAY_START, t1=DAY_END))
    arrivals = len(arribos_set)

    delays: List[float] = []
    landed_prev: Set[int] = set()
    minutos_av = 0
    minutos_av_cong = 0

    for t in range(DAY_START, DAY_END):
        ctrl.step(t, aparicion=(t in arribos_set))

        #Ahora que itero sobre los minutos del dia voy calculando las metricas para un dia

        #CONGESTION: minutos_av con v < vmax en estado approach
        #Para cada momento del dia para cada avion si esta en approach sumo 1 minuto y si ademas esta yendo a vel < vmax sumo uno a congested
        for aid in ctrl.activos:
            av = ctrl.planes[aid]
            vmin, vmax = velocidad_por_distancia(av.distancia_nm)
            minutos_av += 1
            if av.velocidad_kts < vmax - 1e-9:
                minutos_av_cong += 1
        


        #DELAY: Para cada tiempo veo los que aterrizaron en tiempo t y calculo cuanto tardo comparado a su baseline time
        ya_aterrizaron = _landed_set(ctrl)
        recien_aterrizados = ya_aterrizaron - landed_prev
        for aid in recien_aterrizados:
            av = ctrl.planes[aid]

            llegada_esperada = av.aparicion_min + BASELINE_TIME_MIN
            delays.append(t - llegada_esperada)
        
        landed_prev = ya_aterrizaron

    #DIVERTED'S, LANDED'S, DELAY PROMEDIO, CONGESTION RATE
    diverted = len(_diverted_set(ctrl))
    landed = len(landed_prev)
    avg_delay = float(np.mean(delays)) if delays else float('nan')
    cong_rate = (minutos_av_cong / minutos_av) if minutos_av > 0 else float('nan')

    return Metricas(
        lam=lam_per_min,
        arrivals=arrivals,
        landed=landed,
        diverted=diverted,
        avg_delay_min=avg_delay,
        cong_rate=cong_rate,
        delays=delays,
    )
    


#-------------------------------------------------------------------------
# IC95% para una media
#-------------------------------------------------------------------------

def ic95_media(sample: List[float]) -> Tuple[float, float]:
    n = len(sample)
    if n == 0:
        return (float('nan'), float('nan'))
    media = float(np.mean(sample))
    desvio = float(np.std(sample, ddof=1)) if n > 1 else 0.0
    err = desvio / math.sqrt(n) if n > 1 else 0.0
    return (media - 1.96*err, media + 1.96*err)


#-------------------------------------------------------------------------
# MONTECARLO (PARA N DIAS)
#-------------------------------------------------------------------------

def montecarlo_dias(lams: List[float], dias: int = 100, seed: int = 2025,
                    closure_start: int = 180, closure_duration: int = 30) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for lam in lams:
        daily_cong_rates, daily_avg_delays = [], []
        daily_divert_rates, daily_arrivals, daily_diverts = [], [], []

        for _ in range(dias):
            dm = simular_jornada_6(rng.randrange(10**9), lam, closure_start, closure_duration)

            daily_cong_rates.append(dm.cong_rate)
            daily_avg_delays.append(dm.avg_delay_min)

            divert_rate = (dm.diverted / dm.arrivals) if dm.arrivals > 0 else float('nan')
            daily_divert_rates.append(divert_rate)
            daily_arrivals.append(dm.arrivals)
            daily_diverts.append(dm.diverted)

        # agregados + IC95% (igual estilo Ej.4)
        cong_mean = float(np.nanmean(daily_cong_rates))
        cong_ci = ic95_media([x for x in daily_cong_rates if not math.isnan(x)])

        delay_mean = float(np.nanmean(daily_avg_delays))
        delay_ci = ic95_media([x for x in daily_avg_delays if not math.isnan(x)])


        divert_mean = float(np.nanmean(daily_divert_rates))
        divert_ci = ic95_media([x for x in daily_divert_rates if not math.isnan(x)])

        rows.append({
            "lambda_per_min": lam,
            "days": dias,
            "closure_start": closure_start,
            "closure_duration": closure_duration,

            "congestion_rate_mean": cong_mean,
            "congestion_rate_ci_low": cong_ci[0],
            "congestion_rate_ci_high": cong_ci[1],

            "avg_delay_min_mean": delay_mean,
            "avg_delay_min_ci_low": delay_ci[0],
            "avg_delay_min_ci_high": delay_ci[1],

            "divert_rate_mean": divert_mean,
            "divert_rate_ci_low": divert_ci[0],
            "divert_rate_ci_high": divert_ci[1],

            "avg_arrivals_per_day": float(np.nanmean(daily_arrivals)),
            "avg_diverted_per_day": float(np.nanmean(daily_diverts)),
        })

    return pd.DataFrame(rows)


#-------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------


