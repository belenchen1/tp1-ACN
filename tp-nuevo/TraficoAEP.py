
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
        '''
        Hace: reordena la lista de aviones, chequea cada avion y su lider por si tiene que atrasarse, 
        '''
        # pone los nuevos aviones que van a ser turnaroudn y vacia la lista del minuto anterior de los que recien estaban turnaround
        # 
        # para el diccionario de current_speeds de distancias/velocidades
        speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
        dist_prev = {aid: self.planes[aid].distancia_nm for aid in self.activos}
        self.ordenar_activos(current_speeds=speed_prev)

        self.recien_turnaround.clear()
        # Decidir en carril approach
        for aid in list(self.activos):
            av = self.planes[aid]
            vmin, vmax = velocidad_por_distancia(dist_prev[aid])

            leader_id = av.leader_id
            if leader_id is None:
                av.velocidad_kts = vmax
                continue

            my_mins_to_aep = mins_a_aep(dist_prev[aid], speed_prev[aid])
            lead_mins_to_aep = mins_a_aep(dist_prev[leader_id], speed_prev[leader_id])
            gap = my_mins_to_aep - lead_mins_to_aep

            if gap < SEPARACION_PELIGRO:
                nueva_vel = min(vmax, speed_prev[leader_id] - 20.0)
                if nueva_vel < vmin:
                    av.estado = "turnaround"
                    av.velocidad_kts = VEL_TURNAROUND
                    self.mover_a_turnaround(aid)
                    self.recien_turnaround.add(aid)
                else:
                    av.velocidad_kts = max(vmin, nueva_vel)
            else:
                av.velocidad_kts = vmax

        # busco un gap en el que reingresar a los aviones en turnaround en el carril activo/approach
        activos_order = list(self.activos) # lo convierto a lista para no tener problemas si se modifica durante la iteración (es una copia)
        self.intentar_reingreso(activos_order) # en cada step intento reingresar a todos los aviones en turnaround (OJO: no pueden reingresar dos aviones en el mismo step)


    # Reingreso desde turnaround: busca huecos globales y ajusta velocidad
    def intentar_reingreso(self, activos_order: List[int]) -> None:
        if not activos_order: # si no hay ningun avion en el carril de activos --> reingresan todos a vmax (es como darlos vuelta de dirección y pasarlos de carril)
            for aid in list(self.turnaround):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                self.mover_a_activos(aid)
            return

        activos_mins_to_aep = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_mins_to_aep.sort(key=lambda x: x[1])

        for aid in list(self.turnaround):
            av = self.planes[aid]
            d = av.distancia_nm
            vmin, vmax = av.limites_velocidad()
            mins_to_aep_fast = mins_a_aep(d, vmax) # cuanto tardaría si fuese a la velocidad máxima
            mins_to_aep_slow = mins_a_aep(d, vmin) # cuanto tardaría si fuese a la velocidad mínima

            reinsertado = False
            # busco gap entre pares de aviones "consecutivos" (en la lista ordenada por mins_to_aep)
            for (a1, mins_to_aep1), (a2, mins_to_aep2) in zip(activos_mins_to_aep, activos_mins_to_aep[1:]):
                t_low = mins_to_aep1 + SEPARACION_MINIMA # mins que le faltan al anterior + 5 min de separación (minutos minimos en los que podrias llegar)
                t_high = mins_to_aep2 - SEPARACION_MINIMA # mins que le faltan al siguiente - 5 min de separación (minutos maximos en los que podrias llegar)

                if t_high < t_low: # el gap no es suficiente para reingresar
                    continue # sigo buscando entre otros pares

                # si hay suficiente gap para reingresar:
                a = max(t_low, mins_to_aep_fast) # no podes llegar antes del mínimo seguro dada la separacion con el avion de atras, ni más rápido que tu velocidad máxima
                b = min(t_high, mins_to_aep_slow) # no podes llegar después del máximo seguro dada la separacion con el avion de adelante, ni más lento que tu velocidad mínima
                # entonces: [a, b] debería ser un rango factible para reingreso
                if a <= b: # si hay un rango factible, puedo reingresar
                    t_target = (a + b) / 2.0 # me meto en el medio
                    # calculo a qué velocidad tendría que ir para llegar en t_target mins:
                    v_target = (d / t_target) * 60.0 # (d/t_target) es la velocidad  en nm/min, multiplico por 60 para pasar a kts

                    # *CASO BORDE (si el gap es chico, cerca de los 10 min): 
                    # - v_target me podria dar una velocidad fuera de los límites [vmin, vmax]
                        # ej.: si estoy a 100 nm, mis límites son [250, 300] kts
                        # supongo que a y b son [10, 40], entonces t_target=15
                        # v_target = (100 / 15) * 60 = 400 kts -> fuera del limite
                    '''
                    -----|----[-------|------]------|-------- velocidad
                        vt1  vmin    vt2    vmax   vt3
                    vt1 -> voy a vmin. vt2 -> dentro del rango, voy a vt2. vt3 -> voy a vmax
                    '''
                    v_target = min(max(v_target, vmin), vmax) # el max descarta vt1, el min descarta vt3

                    my_mins_to_aep = mins_a_aep(d, v_target)
                    
                    #* CASO BORDE 2 (si el gap es muy chico, cerca de los 10 min):
                    # - al ajustar v_target a los límites, podría salir del rango [t_low, t_high]. por eso vuelvo a chequear.
                    if t_low <= my_mins_to_aep <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        reinsertado = True
                        break
            # si no reingreso en ningún gap entre pares, pruebo antes del primero o después del último:
            # Antes del primero
            if not reinsertado:
                first_mins_to_aep = activos_mins_to_aep[0][1]
                t_high = first_mins_to_aep - SEPARACION_MINIMA
                a = mins_to_aep_fast
                b = min(t_high, mins_to_aep_slow)
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_mins_to_aep = mins_a_aep(d, v_target)
                    if my_mins_to_aep <= t_high: # estoy a + de 5min del primero
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        reinsertado = True
            # Después del último
            if not reinsertado:
                last_mins_to_aep = activos_mins_to_aep[-1][1]
                t_low = last_mins_to_aep + SEPARACION_MINIMA
                a = max(t_low, mins_to_aep_fast)
                b = mins_to_aep_slow
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_mins_to_aep = mins_a_aep(d, v_target)
                    if my_mins_to_aep >= t_low: # estoy a + de 5min del ultimo
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        reinsertado = True
            # si no reinsertó: sigue en turnaround


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
