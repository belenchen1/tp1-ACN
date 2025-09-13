# -*- coding: utf-8 -*-
"""
Simulador de aproximaciones con dos carriles y listas separadas.
Cumple con:
- Lista de vuelos activos (approach)
- Lista de vuelos en turnaround (segundo carril)
- Lista de vuelos inactivos (diverted o landed)
- Cada avión mantiene puntero a su líder (lista simplemente enlazada por ETA)
- IDs únicos e inmutables para mapear aviones
- Reingreso desde turnaround buscando huecos GLOBALES y ajustando velocidad objetivo

Diseño: un TrafficManager orquesta spawns, control y movimiento por pasos.
Unidades: distancia en nm, velocidades en kts, tiempo en minutos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import random

# -----------------------
# Constantes y utilidades
# -----------------------
MINUTE = 1.0
SEPARACION_MINIMA = 5.0  # separación deseada a ambos lados (min)
SEPARACION_PELIGRO = 4.0 # debajo de esto, hay conflicto (actuar fuerte)
VEL_TURNAROUND = 200.0   # kts alejándose
MAX_DIVERTED_DISTANCE = 100.0      # si se aleja más de 100 nm -> diverted

# Bandas (dist_lo, dist_hi, vmin, vmax)
VELOCIDADES = [
    (100.0, float('inf'), 300.0, 500.0),
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

def knots_to_nm_per_min(k: float) -> float:
    return k / 60.0

def velocidad_por_distancia(d_nm: float) -> Tuple[float, float]:
    for lo, hi, vmin, vmax in VELOCIDADES:
        if lo <= d_nm < hi:
            return vmin, vmax
    return VELOCIDADES[0][2], VELOCIDADES[0][3]

def eta_min(dist_nm: float, speed_kts: float) -> float:
    if speed_kts <= 0:
        return float('inf')
    return dist_nm / knots_to_nm_per_min(speed_kts)

# -----------------
# Modelo de Avión
# -----------------
@dataclass
class Avion:
    id: int
    spawn_min: int
    distancia_nm: float = 100.0
    velocidad_kts: float = 300.0
    estado: str = "approach"  # approach | turnaround | diverted | landed
    leader_id: Optional[int] = None  # puntero a su líder en el carril approach

    def limites_velocidad(self) -> Tuple[float, float]:
        return velocidad_por_distancia(self.distancia_nm)

    def eta(self, speed: Optional[float] = None) -> float:
        v = self.velocidad_kts if speed is None else speed
        return eta_min(self.distancia_nm, v)

# ----------------------
# Gestor de Tránsito (FSM)
# ----------------------
class TrafficManager:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.next_id = 1
        self.planes: Dict[int, Avion] = {}
        self.activos: List[int] = []      # ids en estado approach (ordenados por ETA ascendente)
        self.turnaround: List[int] = []    # ids en estado turnaround
        self.inactivos: List[int] = []     # ids en estado diverted | landed
        self._recien_turnaround: Set[int] = set()  # ids que cambiaron a turnaround este paso

    # ---------
    # Creación
    # ---------
    def spawn(self, minuto: int) -> Avion:
        aid = self.next_id
        self.next_id += 1
        av = Avion(id=aid, spawn_min=minuto, distancia_nm=100.0, velocidad_kts=300.0, estado="approach")
        self.planes[aid] = av
        self.activos.append(aid)
        return av

    # --------------------
    # Listas y punteros
    # --------------------
    def _ordenar_activos_por_eta(self, snapshot_speeds: Optional[Dict[int, float]] = None) -> None:
        if not self.activos:
            return
        def eta_of(aid: int) -> float:
            av = self.planes[aid]
            v = av.velocidad_kts if snapshot_speeds is None else snapshot_speeds[aid]
            return av.eta(v)
        self.activos.sort(key=eta_of)
        # actualizar punteros a líder: líder = anterior en ETA (None para el primero)
        for i, aid in enumerate(self.activos):
            av = self.planes[aid]
            av.leader_id = self.activos[i-1] if i > 0 else None

    # --------------------
    # Fase de control
    # --------------------
    def control_paso(self) -> None:
        # Snapshot de distancias/velocidades para decisiones sincrónicas
        speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
        dist_prev = {aid: self.planes[aid].distancia_nm for aid in self.activos}
        self._ordenar_activos_por_eta(snapshot_speeds=speed_prev)

        self._recien_turnaround.clear()
        # Decidir en carril approach
        for aid in list(self.activos):
            av = self.planes[aid]
            vmin, vmax = velocidad_por_distancia(dist_prev[aid])

            leader_id = av.leader_id
            if leader_id is None:
                av.velocidad_kts = vmax
                continue

            my_eta = eta_min(dist_prev[aid], vmax)
            lead_eta = eta_min(dist_prev[leader_id], speed_prev[leader_id])
            gap = my_eta - lead_eta

            if gap < SEPARACION_PELIGRO:
                nueva_vel = min(vmax, speed_prev[leader_id] - 20.0)
                if nueva_vel < vmin:
                    av.estado = "turnaround"
                    av.velocidad_kts = VEL_TURNAROUND
                    self._mover_a_turnaround(aid)
                    self._recien_turnaround.add(aid)
                else:
                    av.velocidad_kts = max(vmin, nueva_vel)
            else:
                av.velocidad_kts = vmax

        # Intentar reingresos con lista de activos (ya ordenada)
        activos_order = list(self.activos)
        self._intentar_reingreso(activos_order)

    def _mover_a_turnaround(self, aid: int) -> None:
        if aid in self.activos:
            self.activos.remove(aid)
        if aid not in self.turnaround:
            self.turnaround.append(aid)
        self.planes[aid].leader_id = None

    def _mover_a_activos(self, aid: int) -> None:
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid not in self.activos:
            self.activos.append(aid)

    def _mover_a_inactivos(self, aid: int) -> None:
        if aid in self.activos:
            self.activos.remove(aid)
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid not in self.inactivos:
            self.inactivos.append(aid)
        self.planes[aid].leader_id = None

    # Reingreso desde turnaround: busca huecos globales y ajusta velocidad para caber en el hueco
    def _intentar_reingreso(self, activos_order: List[int]) -> None:
        if not activos_order:
            # carril vacío: todos reingresan a vmax
            for aid in list(self.turnaround):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                self._mover_a_activos(aid)
            return

        activos_eta = [(aid, self.planes[aid].eta()) for aid in activos_order]
        activos_eta.sort(key=lambda x: x[1])

        for aid in list(self.turnaround):
            av = self.planes[aid]
            d = av.distancia_nm
            vmin, vmax = av.limites_velocidad()
            eta_fast = eta_min(d, vmax)
            eta_slow = eta_min(d, vmin)

            reinsertado = False
            # Intento entre pares consecutivos
            for (a1, eta1), (a2, eta2) in zip(activos_eta, activos_eta[1:]):
                t_low = eta1 + SEPARACION_MINIMA
                t_high = eta2 - SEPARACION_MINIMA
                if t_high < t_low:
                    continue
                a = max(t_low, eta_fast)
                b = min(t_high, eta_slow)
                if a <= b:
                    t_target = 0.5 * (a + b)
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_eta = eta_min(d, v_target)
                    if t_low <= my_eta <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self._mover_a_activos(aid)
                        reinsertado = True
                        break
            # Antes del primero
            if not reinsertado:
                first_eta = activos_eta[0][1]
                t_high = first_eta - SEPARACION_MINIMA
                a = eta_fast
                b = min(t_high, eta_slow)
                if a <= b:
                    t_target = 0.5 * (a + b)
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_eta = eta_min(d, v_target)
                    if my_eta <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self._mover_a_activos(aid)
                        reinsertado = True
            # Después del último
            if not reinsertado:
                last_eta = activos_eta[-1][1]
                t_low = last_eta + SEPARACION_MINIMA
                a = max(t_low, eta_fast)
                b = eta_slow
                if a <= b:
                    t_target = 0.5 * (a + b)
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_eta = eta_min(d, v_target)
                    if my_eta >= t_low:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self._mover_a_activos(aid)
                        reinsertado = True
            # si no reinsertó: sigue en turnaround

    # --------------------
    # Fase de movimiento
    # --------------------
    def move_paso(self) -> None:
        # approach
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * MINUTE
            av.distancia_nm = max(0.0, av.distancia_nm - avance_nm)
            if av.distancia_nm <= 0.0:
                av.distancia_nm = 0.0
                av.velocidad_kts = 0.0
                av.estado = "landed"
                self._mover_a_inactivos(aid)
        # turnaround
        for aid in list(self.turnaround):
            av = self.planes[aid]
            if aid in self._recien_turnaround:
                # no mover en el mismo paso del cambio a turnaround
                continue
            retro_nm = knots_to_nm_per_min(VEL_TURNAROUND) * MINUTE
            av.distancia_nm += retro_nm
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self._mover_a_inactivos(aid)
        # recalcular orden y líderes para próximo paso
        self._ordenar_activos_por_eta()

    # --------------
    # API de simulación
    # --------------
    def bernoulli_spawn_minutes(self, lam_per_min: float, t0: int, t1: int) -> List[int]:
        out = []
        for t in range(t0, t1):
            if self.rng.random() < lam_per_min:
                out.append(t)
        return out

    def step(self, minuto: int, spawn: bool) -> None:
        if spawn:
            self.spawn(minuto)
        self.control_paso()
        self.move_paso()

# ----------------------
# Pequeño runner de prueba (consola)
# ----------------------
if __name__ == "__main__":
    lam = 0.05
    warmup = 60
    t_obs = 180

    tm = TrafficManager(seed=42)
    spawns = set(tm.bernoulli_spawn_minutes(lam, -warmup, t_obs))

    for t in range(-warmup, t_obs):
        tm.step(t, spawn=(t in spawns))
        if t >= 0 and t % 30 == 0:
            print(f"t={t:3d} | activos={len(tm.activos)} turn={len(tm.turnaround)} inactivos={len(tm.inactivos)}")

    landed = sum(1 for i in tm.inactivos if tm.planes[i].estado == "landed")
    diverted = sum(1 for i in tm.inactivos if tm.planes[i].estado == "diverted")
    print(f"Final: landed={landed}, diverted={diverted}, activos={len(tm.activos)}, turn={len(tm.turnaround)}")
