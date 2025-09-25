
#Aviones.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from Helpers import velocidad_por_distancia, mins_a_aep

@dataclass
class Avion:
    id: int
    aparicion_min: int # minuto en el que aparece
    distancia_nm: float = 100.0
    velocidad_kts: float = 300.0
    estado: str = "approach"  # approach | turnaround | diverted | landed
    leader_id: Optional[int] = None  # puntero a su líder en el carril approach (como si fuese una lista simplemente enlazada)
    aterrizaje_min: Optional[int] = None #minuto de aterrizaje
    aterrizaje_min_continuo: Optional [float] = None

    def limites_velocidad(self) -> Tuple[float, float]:
        return velocidad_por_distancia(self.distancia_nm)

    def tiempo_a_aep(self, speed: Optional[float] = None) -> float:
        v = self.velocidad_kts if speed is None else speed
        return mins_a_aep(self.distancia_nm, v)