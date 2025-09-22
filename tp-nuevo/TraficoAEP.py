
#TraficoAEP.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set
import random

from Aviones import Avion
from Constants import (
    MINUTE, SEPARACION_MINIMA, SEPARACION_PELIGRO, VEL_TURNAROUND,
    MAX_DIVERTED_DISTANCE, DAY_END, DAY_START
)
from Helpers import velocidad_por_distancia, knots_to_nm_per_min, mins_a_aep


class TraficoAviones:
    
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed) # genera num aleatorios para apariciones
        self.next_id = 1 # próximo id a asignar (va incrementándose cada vez que aparece un avión)
        self.planes: Dict[int, Avion] = {} # mapeo id -> Avion para acceder más rápido (sin tener que recorrer la lista)
        self.activos: List[int] = []      # ids en estado approach (ordenados por mins_to_aep ascendente)
        self.turnaround: List[int] = []    # ids en estado turnaround
        self.inactivos: List[int] = []     # ids en estado diverted | landed
        self.recien_turnaround: Set[int] = set()  # ids de aviones que cambiaron a turnaround este paso (esto es para que no retrocedan en el mismo paso que cambian de estado)
        self.current_min: int = 0


    #----------------- altas / bajas -----------------------------

    def aparicion(self, minuto: int) -> Avion:
        ''' crear un avion'''
        aid = self.next_id # id que le asigno al avion que apareció
        self.next_id += 1 # actualizo para el próximo avión que aparezca
        av = Avion(id=aid, aparicion_min=minuto, distancia_nm=100.0, velocidad_kts=300.0, estado="approach")
        av.goaround_checked = False 
        self.planes[aid] = av
        self.activos.append(aid) 
        return av


    def mover_a_turnaround(self, aid: int) -> None:
        ''' mueve un avión del carril activo al carril turnaround '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid not in self.turnaround:
            self.turnaround.append(aid)
        self.planes[aid].leader_id = None # ya no tiene líder en el carril approach


    def mover_a_activos(self, aid: int) -> None:
        ''' mueve un avión del carril turnaround al carril activo cuando reingresa '''
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid not in self.activos:
            self.activos.append(aid)


    def mover_a_inactivos(self, aid: int) -> None:
        ''' mueve un avión del carril activo o turnaround a inactivos (landed o diverted) '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid not in self.inactivos:
            self.inactivos.append(aid)
        self.planes[aid].leader_id = None


    #----------------- orden / control ---------------------------------

    def ordenar_activos(self, current_speeds: Optional[Dict[int, float]] = None):
        '''
        current_speeds es un diccionario con una "foto" de las velocidades actuales de los aviones (id -> velocidad_kts).
        '''
        if not self.activos:
            return
        def tiempo_estimado(aid: int) -> float:
            av = self.planes[aid]
            v = av.velocidad_kts if current_speeds is None else current_speeds[aid]
            return av.tiempo_a_aep(v)
        # ordena los activos por tiempo de llegada a aep
        self.activos.sort(key=tiempo_estimado)
        # actualizar punteros a líder: líder = anterior en mins_to_aep (None para el primero)
        for i, aid in enumerate(self.activos):
            av = self.planes[aid]
            av.leader_id = self.activos[i-1] if i > 0 else None


    def control_paso(self) -> None:
        """
        Enforcea SEPARACION_MINIMA con líder por DISTANCIA (no por ETA).
        Barrido Gauss-Seidel con guardrails numéricos:
        - EPS más holgado (1e-6) para evitar micro-oscilaciones.
        - MAX_LOOPS para no colgarse jamás.
        Si para respetar la separación debería ir por debajo de vmin -> turnaround del SEGUIDOR.
        Luego intenta reingresos (actualizando los huecos tras cada inserción).
        """
        # Foto de inicio del minuto
        speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
        dist_prev  = {aid: self.planes[aid].distancia_nm  for aid in self.activos}

        # Definimos líderes por DISTANCIA (más cerca primero)
        order_dist = sorted(self.activos, key=lambda a: dist_prev[a])
        for i, aid in enumerate(order_dist):
            self.planes[aid].leader_id = order_dist[i-1] if i > 0 else None

        self.recien_turnaround.clear()

        EPS = 1e-6       # más holgado para cortar “zig-zag” numérico
        MAX_LOOPS = 50   # guardrail duro
        loops = 0

        while True:
            loops += 1
            if not order_dist or loops > MAX_LOOPS:
                break

            # Velocidades corrientes (arranca con snapshot o las últimas ajustadas)
            v_curr = {aid: speed_prev.get(aid, self.planes[aid].velocidad_kts) for aid in order_dist}
            changed_turn = False
            changed_speed = False

            for i, aid in enumerate(order_dist):
                av = self.planes[aid]
                d  = dist_prev[aid]
                vmin, vmax = velocidad_por_distancia(d)

                if i == 0:
                    # El más cercano va libre a vmax
                    if abs(v_curr.get(aid, 0.0) - vmax) > EPS:
                        v_curr[aid] = vmax
                        av.velocidad_kts = vmax
                        changed_speed = True
                    continue

                leader_id = order_dist[i-1]  # líder = inmediatamente más cercano por DISTANCIA
                if leader_id not in dist_prev:
                    changed_turn = True
                    break

                v_lead  = max(EPS, v_curr.get(leader_id, speed_prev.get(leader_id, self.planes[leader_id].velocidad_kts)))
                lead_eta = mins_a_aep(dist_prev[leader_id], v_lead)

                my_v   = max(EPS, v_curr.get(aid, speed_prev.get(aid, self.planes[aid].velocidad_kts)))
                my_eta = mins_a_aep(d, my_v)

                min_eta = lead_eta + SEPARACION_MINIMA
                if my_eta < (min_eta - EPS):
                    # Necesito demorarme: apunto a llegar justo a min_eta
                    needed_v = (d / max(EPS, min_eta)) * 60.0
                    if needed_v < (vmin - EPS):
                        # No alcanza bajando: el SEGUIDOR (el de atrás por DISTANCIA) a turnaround
                        av.estado = "turnaround"
                        av.velocidad_kts = VEL_TURNAROUND
                        self.mover_a_turnaround(aid)
                        self.recien_turnaround.add(aid)
                        changed_turn = True
                        break
                    target_v = min(max(needed_v, vmin), vmax)
                else:
                    # Voy libre a vmax (no rompo separación)
                    target_v = vmax

                if abs(target_v - v_curr.get(aid, my_v)) > EPS:
                    v_curr[aid] = target_v
                    av.velocidad_kts = target_v
                    changed_speed = True

            if changed_turn:
                # Alguien cambió de carril: refrescar fotos y rearmar orden por DISTANCIA
                speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
                dist_prev  = {aid: self.planes[aid].distancia_nm  for aid in self.activos}
                order_dist = sorted(self.activos, key=lambda a: dist_prev[a])
                for i, aid in enumerate(order_dist):
                    self.planes[aid].leader_id = order_dist[i-1] if i > 0 else None
                continue

            if not changed_speed:
                break

            # Propagar los ajustes de esta pasada a la “foto” para la próxima iteración
            speed_prev = {aid: v_curr.get(aid, self.planes[aid].velocidad_kts) for aid in order_dist}

        # Reingreso (greedy), actualizando huecos tras cada inserción
        activos_order = list(self.activos)
        self.intentar_reingreso(activos_order)




    def intentar_reingreso(self, activos_order: List[int]) -> None:
        """
        Inserta TODOS los turnaround que entren en huecos válidos.
        Tras cada inserción actualiza la cola de ETAs (no mete dos en el mismo hueco).
        """
        # Si no hay activos, vuelven todos directo a approach @ vmax
        if not activos_order:
            for aid in list(self.turnaround):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                self.mover_a_activos(aid)
            return

        # Construyo lista (id, ETA) ordenada
        activos_mins = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_mins.sort(key=lambda x: x[1])

        EPS = 1e-9

        while True:
            inserted_any = False
            for aid in list(self.turnaround):
                if aid not in self.turnaround:
                    continue  # pudo reinsertarse en una vuelta anterior

                av = self.planes[aid]
                d = av.distancia_nm
                vmin, vmax = av.limites_velocidad()
                t_fast = mins_a_aep(d, vmax)
                t_slow = mins_a_aep(d, vmin)

                placed = False

                # 1) Entre pares
                for (a1, t1), (a2, t2) in zip(activos_mins, activos_mins[1:]):
                    t_low  = t1 + SEPARACION_MINIMA
                    t_high = t2 - SEPARACION_MINIMA
                    if t_high + EPS < t_low:
                        continue
                    a = max(t_low,  t_fast)
                    b = min(t_high, t_slow)
                    if a - EPS <= b:
                        t_target = 0.5 * (a + b)
                        v_target = (d / t_target) * 60.0
                        v_target = min(max(v_target, vmin), vmax)
                        my_t = mins_a_aep(d, v_target)
                        if t_low - EPS <= my_t <= t_high + EPS:
                            av.velocidad_kts = v_target
                            av.estado = "approach"
                            self.mover_a_activos(aid)
                            # actualizar agenda
                            activos_mins.append((aid, my_t))
                            activos_mins.sort(key=lambda x: x[1])
                            inserted_any = True
                            placed = True
                            break
                if placed:
                    continue

                # 2) Antes del primero
                first_t = activos_mins[0][1]
                t_high = first_t - SEPARACION_MINIMA
                a = t_fast
                b = min(t_high, t_slow)
                if a - EPS <= b:
                    t_target = 0.5 * (a + b)
                    v_target = (d / t_target) * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_t = mins_a_aep(d, v_target)
                    if my_t <= t_high + EPS:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        activos_mins.append((aid, my_t))
                        activos_mins.sort(key=lambda x: x[1])
                        inserted_any = True
                        continue

                # 3) Después del último
                last_t = activos_mins[-1][1]
                t_low = last_t + SEPARACION_MINIMA
                a = max(t_low, t_fast)
                b = t_slow
                if a - EPS <= b:
                    t_target = 0.5 * (a + b)
                    v_target = (d / t_target) * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_t = mins_a_aep(d, v_target)
                    if my_t + EPS >= t_low:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        activos_mins.append((aid, my_t))
                        activos_mins.sort(key=lambda x: x[1])
                        inserted_any = True
                        continue

            if not inserted_any:
                break



    #-------------------------- mover --------------------------------

    def mover_paso(self) -> None:
        # approach
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * MINUTE
            d_prev = av.distancia_nm
            new_dist = max(0.0, d_prev - avance_nm)
            if new_dist <= 0.0: # llegó a aep
                s = (d_prev / avance_nm) if avance_nm > 0 else 1.0
                av.aterrizaje_min = self.current_min
                av.aterrizaje_min_continuo = float(self.current_min) + s

                av.distancia_nm = 0.0
                av.velocidad_kts = 0.0
                av.estado = "landed"
                self.mover_a_inactivos(aid)
                continue 
            av.distancia_nm = new_dist
            
        # turnaround
        for aid in list(self.turnaround):
            av = self.planes[aid]
            if aid in self.recien_turnaround:
                # no mover en el mismo paso del cambio a turnaround
                continue
            retro_nm = knots_to_nm_per_min(VEL_TURNAROUND) * MINUTE
            av.distancia_nm += retro_nm
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)
        # recalcular orden y líderes para próximo paso
        self.ordenar_activos()


    #------------------------------ apariciones / step --------------------------

    def bernoulli_aparicion(self, lam_per_min: float, t0: int = DAY_START, t1: int = DAY_END) -> List[int]:
        '''
        devuelve una lista de los t's en los que aparecen los aviones
        la bernoulli es una aproximación de la forma discreta al proceso de Poisson
        '''
        apariciones = []
        for t in range(t0, t1):
            if self.rng.random() < lam_per_min:
                apariciones.append(t)
        return apariciones


    def step(self, minuto: int, aparicion: bool) -> None:
        self.current_min = minuto
        if aparicion:
            # crea el avion
            self.aparicion(minuto)
        # controla (cambia de carril, actualiza, etc) y mueve todos los aviones un paso (1 minuto)
        self.control_paso()
        self.mover_paso()
