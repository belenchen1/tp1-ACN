# TraficoAEPCierre.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

from TraficoAEP import TraficoAviones
from Constants import (
    MINUTE, SEPARACION_MINIMA, SEPARACION_PELIGRO, VEL_TURNAROUND,
    MAX_DIVERTED_DISTANCE
)
from Helpers import knots_to_nm_per_min, mins_a_aep, velocidad_por_distancia


@dataclass
class AEPCerrado:
    start_min: int = 20
    dur_min: int = 30

    def is_closed(self, t: int) -> bool:
        return self.start_min <= t < (self.start_min + self.dur_min)

    @property
    def reopen_min(self) -> int:
        return self.start_min + self.dur_min


class TraficoAEPCerrado(TraficoAviones):
    """
    Cierre sorpresivo de AEP por 30 min (sin avión bloqueando en 0 nm).

    - Si AEP está cerrado:
        * Activos cuyo ETA (a vmax, snapshot) < tiempo restante -> turnaround inmediato.
        * Aviones que APARECEN se evalúan en el acto: si a vmax llegan antes de reabrir, van directo a turnaround.
        * Reingreso desde turnaround solo si, aun a vmax, llegarían luego de la reapertura.
    - Separaciones: líder por DISTANCIA (no por ETA). Sólo el SEGUIDOR puede ir a turnaround.
    - Guardrails numéricos para evitar oscilaciones/locks.
    """

    def __init__(self, closure: AEPCerrado, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closure = closure
        self.current_minute: int = 0
        self.appear_time: Dict[int, int] = {}
        self.land_time: Dict[int, int] = {}

    # ---------------- Altas ----------------

    def aparicion(self, minuto: int):
        """Crear avión y, si está cerrado, decidir si va directo a turnaround."""
        av = super().aparicion(minuto)  # approach @100nm, 300kts
        self.appear_time[av.id] = minuto

        if self.closure.is_closed(minuto):
            vmin, vmax = av.limites_velocidad()
            eta_fast = mins_a_aep(av.distancia_nm, vmax)
            # Si aterrizaría antes de reabrir -> no entra a la cola, se va a turnaround.
            if minuto + eta_fast < self.closure.reopen_min:
                av.estado = "turnaround"
                av.velocidad_kts = VEL_TURNAROUND
                self.mover_a_turnaround(av.id)
                self.recien_turnaround.add(av.id)

        return av

    # --------------- Control (con política de cierre) ---------------

    def control_paso(self) -> None:
        """
        1) Si está cerrado, manda a turnaround los activos que llegan antes de reabrir (usando ETA a vmax).
        2) Enforcea separación mínima con líder por DISTANCIA via Gauss–Seidel (sólo el seguidor puede ir a turnaround).
        3) Reingresa turnarounds (posiblemente varios en el mismo step), respetando cierre y separaciones.
        """
        EPS = 1e-6
        MAX_LOOPS = 50

        # Foto de inicio del minuto
        speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
        dist_prev  = {aid: self.planes[aid].distancia_nm  for aid in self.activos}

        self.recien_turnaround.clear()

        # (1) Turnaround forzado durante cierre si el ETA (a vmax) cae dentro de la ventana restante.
        if self.closure.is_closed(self.current_minute):
            remaining = self.closure.reopen_min - self.current_minute
            for aid in list(self.activos):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                eta_fast = mins_a_aep(dist_prev[aid], vmax)  # ETA con vmax (snapshot)
                if eta_fast < remaining - EPS:
                    av.estado = "turnaround"
                    av.velocidad_kts = VEL_TURNAROUND
                    self.mover_a_turnaround(aid)
                    self.recien_turnaround.add(aid)

            # Refrescar snapshot luego de las bajas
            speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
            dist_prev  = {aid: self.planes[aid].distancia_nm  for aid in self.activos}

        # (2) Separaciones con líder por DISTANCIA (no por ETA), vía Gauss–Seidel
        # Definir orden por DISTANCIA (más cerca primero) y leader_id coherente
        order_dist = sorted(self.activos, key=lambda a: dist_prev[a]) if self.activos else []
        for i, aid in enumerate(order_dist):
            self.planes[aid].leader_id = order_dist[i-1] if i > 0 else None

        loops = 0
        while True:
            loops += 1
            if not order_dist or loops > MAX_LOOPS:
                break

            # Velocidades corrientes (arrancan en snapshot / última iteración)
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
                    # Necesito demorarme para llegar justo a min_eta
                    needed_v = (d / max(EPS, min_eta)) * 60.0
                    if needed_v < (vmin - EPS):
                        # No alcanza bajando: el SEGUIDOR (detrás por DISTANCIA) se va a turnaround
                        av.estado = "turnaround"
                        av.velocidad_kts = VEL_TURNAROUND
                        self.mover_a_turnaround(aid)
                        self.recien_turnaround.add(aid)
                        changed_turn = True
                        break
                    target_v = min(max(needed_v, vmin), vmax)
                else:
                    # Voy libre a vmax
                    target_v = vmax

                if abs(target_v - v_curr.get(aid, my_v)) > EPS:
                    v_curr[aid] = target_v
                    av.velocidad_kts = target_v
                    changed_speed = True

            if changed_turn:
                # Alguien cambió de carril: refrescar fotos y rearmar orden por DISTANCIA
                speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
                dist_prev  = {aid: self.planes[aid].distancia_nm  for aid in self.activos}
                order_dist = sorted(self.activos, key=lambda a: dist_prev[a]) if self.activos else []
                for i, aid in enumerate(order_dist):
                    self.planes[aid].leader_id = order_dist[i-1] if i > 0 else None
                continue

            if not changed_speed:
                break

            # Propagar los ajustes de esta pasada a la “foto” para la próxima iter
            speed_prev = {aid: v_curr.get(aid, self.planes[aid].velocidad_kts) for aid in order_dist}

        # (3) Reingreso de turnarounds, potencialmente varios en el mismo step
        self.intentar_reingreso_cierre()

    # --------------- Reingreso con restricción de reapertura ---------------

    def intentar_reingreso_cierre(self) -> None:
        """
        Intenta reinsertar TODOS los turnarounds en el mismo step.
        Cada vez que uno entra, se recalcula la cola de activos y se vuelve a intentar con el resto.
        Durante cierre, solo se reinsertan si (current + mins_fast) >= reopen_min.
        """
        EPS = 1e-6

        while True:
            changed = False

            if not self.activos:
                # Si no hay activos, pueden volver directo… salvo que el cierre impida llegar luego de reabrir.
                for aid in list(self.turnaround):
                    av = self.planes[aid]
                    vmin, vmax = av.limites_velocidad()
                    t_fast = mins_a_aep(av.distancia_nm, vmax)
                    if (not self.closure.is_closed(self.current_minute)) or \
                       (self.current_minute + t_fast >= self.closure.reopen_min - EPS):
                        av.velocidad_kts = vmax
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        changed = True
                if not changed:
                    break
                else:
                    continue  # recalcular en la próxima vuelta

            # Construir lista de tiempos de los activos actual (ordenada)
            activos_mins = [(aid, self.planes[aid].tiempo_a_aep()) for aid in self.activos]
            activos_mins.sort(key=lambda x: x[1])

            cerrado = self.closure.is_closed(self.current_minute)
            reopen_min = self.closure.reopen_min

            # Probar todos los turnarounds
            for aid in list(self.turnaround):
                av = self.planes[aid]
                d = av.distancia_nm
                vmin, vmax = av.limites_velocidad()
                mins_fast = mins_a_aep(d, vmax)  # “mejor caso” para chequear contra reapertura
                mins_slow = mins_a_aep(d, vmin)

                # Restricción de cierre: si incluso a vmax llegaría antes de reabrir, ni intento.
                if cerrado and (self.current_minute + mins_fast < reopen_min - EPS):
                    continue

                reinsertado = False

                # ---- Intentar entre pares
                for (a1, t1), (a2, t2) in zip(activos_mins, activos_mins[1:]):
                    t_low  = t1 + SEPARACION_MINIMA
                    t_high = t2 - SEPARACION_MINIMA
                    if t_high + EPS < t_low:
                        continue
                    a = max(t_low, mins_fast)
                    b = min(t_high, mins_slow)
                    if a - EPS <= b:
                        t_target = 0.5 * (a + b)
                        v_target = min(max((d / t_target) * 60.0, vmin), vmax)
                        my_mins  = mins_a_aep(d, v_target)
                        if t_low - EPS <= my_mins <= t_high + EPS:
                            av.velocidad_kts = v_target
                            av.estado = "approach"
                            self.mover_a_activos(aid)
                            # Recalcular cola para próximos intentos
                            activos_mins = [(x, self.planes[x].tiempo_a_aep()) for x in self.activos]
                            activos_mins.sort(key=lambda x: x[1])
                            changed = True
                            reinsertado = True
                            break
                if reinsertado:
                    continue

                # ---- Intentar antes del primero
                first_t = activos_mins[0][1]
                t_high = first_t - SEPARACION_MINIMA
                a = mins_fast
                b = min(t_high, mins_slow)
                if a - EPS <= b:
                    t_target = 0.5 * (a + b)
                    v_target = min(max((d / t_target) * 60.0, vmin), vmax)
                    my_mins  = mins_a_aep(d, v_target)
                    if my_mins <= t_high + EPS:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        activos_mins = [(x, self.planes[x].tiempo_a_aep()) for x in self.activos]
                        activos_mins.sort(key=lambda x: x[1])
                        changed = True
                        continue

                # ---- Intentar después del último
                last_t = activos_mins[-1][1]
                t_low = last_t + SEPARACION_MINIMA
                a = max(t_low, mins_fast)
                b = mins_slow
                if a - EPS <= b:
                    t_target = 0.5 * (a + b)
                    v_target = min(max((d / t_target) * 60.0, vmin), vmax)
                    my_mins  = mins_a_aep(d, v_target)
                    if my_mins + EPS >= t_low:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        activos_mins = [(x, self.planes[x].tiempo_a_aep()) for x in self.activos]
                        activos_mins.sort(key=lambda x: x[1])
                        changed = True
                        continue

            if not changed:
                break  # no hubo más reinserciones en esta vuelta

    # ---------------- Movimiento ----------------

    def mover_paso(self) -> None:
        # APPROACH
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * MINUTE
            d_prev = av.distancia_nm
            new_dist = max(0.0, d_prev - avance_nm)
            would_land = (new_dist <= 0.0)

            # Durante cierre, nadie aterriza: si “llegaba”, lo mando a turnaround.
            if would_land and self.closure.is_closed(self.current_minute):
                av.estado = "turnaround"
                av.velocidad_kts = VEL_TURNAROUND
                av.distancia_nm = max(d_prev, 0.01)  # evitar 0 exacto
                self.mover_a_turnaround(aid)
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

        # TURNAROUND (igual a base)
        for aid in list(self.turnaround):
            av = self.planes[aid]
            if aid in self.recien_turnaround:
                continue
            av.distancia_nm += knots_to_nm_per_min(VEL_TURNAROUND) * MINUTE
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)

        self.ordenar_activos()

    # ---------------- Step ----------------

    def step(self, minuto: int, aparicion: bool) -> None:
        self.current_minute = minuto
        if aparicion:
            self.aparicion(minuto)
        self.control_paso()
        self.mover_paso()
