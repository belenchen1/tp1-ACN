
#TraficoAEPViento.py

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from TraficoAEP import TraficoAviones
from Aviones import Avion
from Constants import (
    MINUTE, SEPARACION_MINIMA, SEPARACION_PELIGRO, VEL_TURNAROUND,
    MAX_DIVERTED_DISTANCE, DAY_END, DAY_START
)

from Helpers import velocidad_por_distancia, mins_a_aep, knots_to_nm_per_min



class TraficoAEPViento(TraficoAviones):

    # Suponiendo que vaya a 150 nudos en el final, 150 nudos = 150 millas náuticas/hora → 150÷60= 2,5 millas náuticas por minuto
    def __init__(self, seed: int = 42, p_goaround: float = 0.10, final_threshold_nm: float = 2.5) -> None:
        super().__init__(seed=seed)
        self.p_goaround = float(p_goaround)
        self.final_threshold_nm = float(final_threshold_nm)
        # estado adicional
        self.interrupted: List[int] = []
        self.recien_interrupted: Set[int] = set()
        self.current_min: int = DAY_START


    #------------------ helpers adicionales --------------------------

    def mover_a_interrupted(self, aid: int) -> None:
        if aid in self.activos:
            self.activos.remove(aid)
        if aid not in self.interrupted:
            self.interrupted.append(aid)
            self.recien_interrupted.add(aid)
        self.planes[aid].leader_id = None


    def mover_interrupted_a_activos(self, aid: int) -> None:
        if aid in self.interrupted:
            self.interrupted.remove(aid)
        if aid not in self.activos:
            self.activos.append(aid)

    def mover_a_inactivos(self, aid: int) -> None:
        ''' mueve un avión del carril activo o turnaround o interrupted a inactivos (landed o diverted) '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid in self.interrupted:
            self.interrupted.remove(aid)
        if aid not in self.inactivos:
            self.inactivos.append(aid)
        self.planes[aid].leader_id = None


    # ---------- control ----------

    def control_paso(self) -> None:
        # snapshot de velocidades y distancias antes de decidir
        speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
        dist_prev = {aid: self.planes[aid].distancia_nm for aid in self.activos}
        self.ordenar_activos(current_speeds=speed_prev)

        self.recien_turnaround.clear()
        self.recien_interrupted.clear()

        for aid in list(self.activos):
            av = self.planes[aid]
            vmin, vmax = velocidad_por_distancia(dist_prev[aid])

            # --- go-around por viento: sólo si aterrizaba en este paso ---
            avance_nm = knots_to_nm_per_min(speed_prev[aid]) * MINUTE
            aterriza_este_paso = (dist_prev[aid] - avance_nm) <= 0.0

            if aterriza_este_paso:
                # tirar la moneda una única vez por avión
                if not getattr(av, "goaround_checked", False):
                    av.goaround_checked = True
                    if self.rng.random() < self.p_goaround:
                        av.estado = "interrupted"
                        av.velocidad_kts = VEL_TURNAROUND
                        self.mover_a_interrupted(aid)
                        continue 


            # --- control normal (separación con líder) ---
            leader_id = av.leader_id
            if leader_id is None:
                av.velocidad_kts = vmax
                continue

            # my_eta = mins_a_aep(dist_prev[aid], vmax)
            my_eta = mins_a_aep(dist_prev[aid], speed_prev[aid])
            lead_eta = mins_a_aep(dist_prev[leader_id], speed_prev[leader_id])
            gap = my_eta - lead_eta

            if gap < SEPARACION_PELIGRO:
                nueva = min(vmax, speed_prev[leader_id] - 20.0)
                if nueva < vmin:
                    av.estado = "turnaround"
                    av.velocidad_kts = VEL_TURNAROUND
                    self.mover_a_turnaround(aid)
                    self.recien_turnaround.add(aid)
                else:
                    av.velocidad_kts = max(vmin, nueva)
            else:
                av.velocidad_kts = vmax

        # reingreso: probar huecos para turnaround + interrupted
        activos_order = list(self.activos)
        self.intentar_reingreso_con_interrupted(activos_order)


        







    def intentar_reingreso_con_interrupted(self, activos_order: List[int]) -> None:
        """
        Igual que intentar_reingreso(base), pero permite reinsertar también
        los 'interrupted' (go-around) con la misma regla de huecos.
        Extra: por seguridad, no reinsertar interrupted si d <= 5 nm.
        """
        if not activos_order:
            # si no hay activos, vuelven directo a approach a vmax
            for aid in list(self.turnaround) + list(self.interrupted):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                if aid in self.turnaround:
                    self.mover_a_activos(aid)
                elif aid in self.interrupted:
                    self.mover_interrupted_a_activos(aid)
            return

        # (id, ETA) ordenado por llegada
        activos_eta = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_eta.sort(key=lambda x: x[1])

        reintegrables = list(self.turnaround) + list(self.interrupted)
        for aid in reintegrables:
            av = self.planes[aid]
            d = av.distancia_nm
            vmin, vmax = av.limites_velocidad()
            t_fast = mins_a_aep(d, vmax)
            t_slow = mins_a_aep(d, vmin)

            # restricción para interrupted: no reinsertar si ya está pegado a AEP
            if aid in self.interrupted and d <= 5.0:
                continue

            reinsertado = False

            # entre pares
            for (a1, t1), (a2, t2) in zip(activos_eta, activos_eta[1:]):
                t_low = t1 + SEPARACION_MINIMA
                t_high = t2 - SEPARACION_MINIMA
                if t_high < t_low:
                    continue
                a = max(t_low, t_fast)
                b = min(t_high, t_slow)
                if a <= b:
                    t_target = 0.5 * (a + b)
                    v_target = (d / t_target) * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_mins_to_aep = mins_a_aep(d, v_target)
                    if t_low <= my_mins_to_aep <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        if aid in self.turnaround:
                            self.mover_a_activos(aid)
                        else:
                            self.mover_interrupted_a_activos(aid)
                        reinsertado = True
                        break
            if reinsertado:
                continue

            # antes del primero
            first_t = activos_eta[0][1]
            t_high = first_t - SEPARACION_MINIMA
            a = t_fast
            b = min(t_high, t_slow)
            if a <= b:
                t_target = 0.5 * (a + b)
                v_target = (d / t_target) * 60.0
                v_target = min(max(v_target, vmin), vmax)
                my_mins_to_aep = mins_a_aep(d, v_target)
                if my_mins_to_aep <= t_high:
                    av.velocidad_kts = v_target
                    av.estado = "approach"
                    if aid in self.turnaround:
                        self.mover_a_activos(aid)
                    else:
                        self.mover_interrupted_a_activos(aid)
                    continue

            # después del último
            last_t = activos_eta[-1][1]
            t_low = last_t + SEPARACION_MINIMA
            a = max(t_low, t_fast)
            b = t_slow
            if a <= b:
                t_target = 0.5 * (a + b)
                v_target = (d / t_target) * 60.0
                v_target = min(max(v_target, vmin), vmax)
                my_mins_to_aep = mins_a_aep(d, v_target)
                if my_mins_to_aep >= t_low:
                    av.velocidad_kts = v_target
                    av.estado = "approach"
                    if aid in self.turnaround:
                        self.mover_a_activos(aid)
                    else:
                        self.mover_interrupted_a_activos(aid)
                    continue



    """ def intentar_reingreso_con_interrupted(self, activos_order: List[int]) -> None:
        
        Igual que intentar_reingreso(base), pero permite reinsertar también
        los 'interrupted' (go-around) con la misma regla de huecos.
        Extra: por seguridad, no reinsertar interrupted si d <= 5 nm.
        
        if not activos_order:
            # si no hay activos, vuelven directo a approach a vmax
            for aid in list(self.turnaround) + list(self.interrupted):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                if aid in self.turnaround:
                    self.mover_a_activos(aid)
                elif aid in self.interrupted:
                    self.mover_interrupted_a_activos(aid)
            return

        # (id, ETA) ordenado por llegada
        activos_eta = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_eta.sort(key=lambda x: x[1])

        reintegrables = list(self.turnaround) + list(self.interrupted)
        for aid in reintegrables:
            av = self.planes[aid]
            d = av.distancia_nm
            vmin, vmax = av.limites_velocidad()
            t_fast = mins_a_aep(d, vmax)
            t_slow = mins_a_aep(d, vmin)

            # restricción para interrupted: no reinsertar si ya está pegado a AEP
            if aid in self.interrupted and d <= 5.0:
                continue

            reinsertado = False

            # entre pares
            for (a1, t1), (a2, t2) in zip(activos_eta, activos_eta[1:]):
                t_low = t1 + SEPARACION_MINIMA
                t_high = t2 - SEPARACION_MINIMA
                if t_high < t_low:
                    continue
                a = max(t_low, t_fast)
                b = min(t_high, t_slow)
                if a <= b:
                    t_target = 0.5 * (a + b)
                    v_target = min(max((d / t_target) * 60.0, vmin), vmax)
                    my_t = mins_a_aep(d, v_target)
                    if t_low <= my_t <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        if aid in self.turnaround:
                            self.mover_a_activos(aid)
                        else:
                            self.mover_interrupted_a_activos(aid)
                        reinsertado = True
                        break
            if reinsertado:
                continue

            # antes del primero
            first_t = activos_eta[0][1]
            t_high = first_t - SEPARACION_MINIMA
            a = t_fast
            b = min(t_high, t_slow)
            if a <= b:
                t_target = 0.5 * (a + b)
                v_target = min(max((d / t_target) * 60.0, vmin), vmax)
                if mins_a_aep(d, v_target) <= t_high:
                    av.velocidad_kts = v_target
                    av.estado = "approach"
                    if aid in self.turnaround:
                        self.mover_a_activos(aid)
                    else:
                        self.mover_interrupted_a_activos(aid)
                    continue

            # después del último
            last_t = activos_eta[-1][1]
            t_low = last_t + SEPARACION_MINIMA
            a = max(t_low, t_fast)
            b = t_slow
            if a <= b:
                t_target = 0.5 * (a + b)
                v_target = min(max((d / t_target) * 60.0, vmin), vmax)
                if mins_a_aep(d, v_target) >= t_low:
                    av.velocidad_kts = v_target
                    av.estado = "approach"
                    if aid in self.turnaround:
                        self.mover_a_activos(aid)
                    else:
                        self.mover_interrupted_a_activos(aid) """



    # ---------- movimiento ----------
    def mover_paso(self) -> None:
        # approach (idéntico a base, setea aterrizaje_min)
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * MINUTE
            d_prev = av.distancia_nm
            new_dist = max(0.0, d_prev - avance_nm)
            if new_dist <= 0.0:
                s = (d_prev/avance_nm) if avance_nm > 0 else 1.0
                av.aterrizaje_min = int(self.current_min)
                t_cont = float(self.current_min) + float(s)
                setattr(self.planes[aid], "aterrizaje_min_cont", float(t_cont))
                setattr(self.planes[aid], "aterrizaje_min_continuo", float(t_cont))
                av.distancia_nm = 0.0
                av.velocidad_kts = 0.0
                av.estado = "landed"
                self.mover_a_inactivos(aid)
                continue

            self.planes[aid].distancia_nm = new_dist

        # turnaround (igual a base)
        for aid in list(self.turnaround):
            if aid in self.recien_turnaround:
                continue
            av = self.planes[aid]
            av.distancia_nm += knots_to_nm_per_min(VEL_TURNAROUND) * MINUTE
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)

        # interrupted (nuevo: se aleja también a VEL_TURNAROUND)
        for aid in list(self.interrupted):
            if aid in self.recien_interrupted:
                continue
            av = self.planes[aid]
            av.distancia_nm += knots_to_nm_per_min(VEL_TURNAROUND) * MINUTE
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)

        self.ordenar_activos()


    # ---------- step ----------
    def step(self, minuto: int, aparicion: bool) -> None:
        self.current_min = minuto
        if aparicion:
            self.aparicion(minuto)
        self.control_paso()
        self.mover_paso()

    def aviones_landed(self) -> List[Avion]:
        return [av for av in self.planes.values() if av.aterrizaje_min is not None]
    # filtra solo los aviones que realmente aterrizaron 