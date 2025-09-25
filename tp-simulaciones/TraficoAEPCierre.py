
#TraficoAEPCierre.py 

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Set, Dict

from TraficoAEP import TraficoAviones
from Constants import (
    MINUTE, SEPARACION_MINIMA, SEPARACION_PELIGRO, VEL_TURNAROUND,
    MAX_DIVERTED_DISTANCE, DAY_END, DAY_START
)
from Helpers import knots_to_nm_per_min


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
        self.current_minute: int = 0
        self.appear_time: Dict[int, int] = {}
        self.land_time: Dict[int, int] = {}
    
    def aparicion(self, minuto:int):
        if not hasattr(self, "appear_time"): #parche de defensa por si no tiene appear time
            self.appear_time = {}

        av = super().aparicion(minuto)
        self.appear_time[av.id] = minuto
        return av
    

    def mover_paso(self) -> None:
       
        #Si hay un bloqueador y REABRIO, lo aterrizo y libero
        if self.blocking_id is not None and not self.closure.is_closed(self.current_minute):
            bid = self.blocking_id
            if bid in self.activos:
                avb = self.planes[bid]
                avb.aterrizaje_min = self.current_minute
                avb.aterrizaje_min_continuo = float(self.current_minute)
                avb.distancia_nm = 0.0
                avb.velocidad_kts = 0.0
                avb.estado = "landed"
                self.mover_a_inactivos(bid)
            self.blocking_id = None
        
        
        #APPROACH
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * MINUTE
            d_prev = av.distancia_nm
            new_dist = max(0.0, d_prev - avance_nm)
            would_land = (new_dist <= 0) #va a ser true si en el siguiente paso le tocaria aterrizar

            #Si en el siguiente paso tendria que aterrizar y AEP esta cerrado:
                #Si no hay un avion ya blockeando se pone como avion que "blockea"
                #Si ya hay un avion blockeando lo pego al que esta blockeando para que en control reaccione segun las restricciones
            #Si no estaba cerrado lo aterrizo normalmente

            if would_land and self.closure.is_closed(self.current_minute):
                if self.blocking_id is None:
                    self.blocking_id = aid
                    av.distancia_nm = 0.0
                    av.velocidad_kts = 0.0
                    av.estado = "approach"
                
                else:
                    av.distancia_nm = max(d_prev, 0.01)
                continue

            if would_land:
                s = (d_prev / avance_nm) if avance_nm > 0 else 1.0
                av.aterrizaje_min = self.current_minute
                av.aterrizaje_min_continuo = float(self.current_minute) + s
                av.distancia_nm = 0.0
                av.velocidad_kts = 0.0
                av.estado = "landed"
                self.mover_a_inactivos(aid)
                continue

            av.distancia_nm = new_dist

    
        #TURNAROUND: va a ser igual que en la clase base
        for aid in list(self.turnaround):
            av = self.planes[aid]
            if aid in self.recien_turnaround:
                continue
            retro_nm = knots_to_nm_per_min(VEL_TURNAROUND) * 1.0
            av.distancia_nm += retro_nm
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)

        
        self.ordenar_activos()

        
    #Se define el paso pero para el caso con AEP cerrado
    def step(self, minuto: int, aparicion: bool) -> None:
        self.current_minute = minuto
        if aparicion:
            self.aparicion(minuto)
        self.control_paso()
        self.mover_paso()