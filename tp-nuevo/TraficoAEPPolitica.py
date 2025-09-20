
# TraficoAEPPolitica.py

from __future__ import annotations
from typing import Dict, List, Optional, Set

from TraficoAEP import TraficoAviones
from Constants import SEPARACION_PELIGRO, VEL_TURNAROUND, BUFFER_ANTICIPACION
from Helpers import velocidad_por_distancia, mins_a_aep, g_objetivo

class TraficoAvionesPolitica(TraficoAviones):
    def __init__(self, seed: int = 42, buffer_anticipacion: float = 0.5) -> None:
        super().__init__(seed=seed)
        self.buffer_anticipacion = float(buffer_anticipacion)


    def control_paso(self) -> None:
        '''
        Reordena y decide velocidades/estados por un paso:
        1) Ordena activos y setea punteros a líder.
        2) Para cada avión en approach:
           - Chequeo de seguridad (gap < SEPARACION_PELIGRO) -> frenar fuerte o pasar a turnaround.
           - Si no hay peligro, aplica Política A (metering):
             apuntar a ETA = ETA_líder + g_objetivo(dist) con límites de banda.
        3) Intenta reinsertar desde turnaround a huecos globales.
        '''
        # Foto de dist/vel previas a las decisiones del paso
        speed_prev = {aid: self.planes[aid].velocidad_kts for aid in self.activos}
        dist_prev  = {aid: self.planes[aid].distancia_nm  for aid in self.activos}
        # Orden consistente usando la foto
        self.ordenar_activos(current_speeds=speed_prev)

        # limpiar marcas de "recién enviados a turnaround" para este paso
        self.recien_turnaround.clear()

        # --- Decidir en carril approach ---
        for aid in list(self.activos):
            av = self.planes[aid]
            vmin, vmax = velocidad_por_distancia(dist_prev[aid])

            leader_id = av.leader_id
            if leader_id is None:
                # Sin líder: nadie delante -> ir a vmax (no hacemos metering al primero)
                av.velocidad_kts = vmax
                continue

            # ETA del líder con su velocidad previa (foto consistente del paso)
            my_mins_to_aep = mins_a_aep(dist_prev[aid], speed_prev[aid])
            lead_mins_to_aep = mins_a_aep(dist_prev[leader_id], speed_prev[leader_id])

            # --- 1) Chequeo de seguridad (fallback original) ---
            # Gap "pesimista" evaluado si YO fuera a vmax (peor caso para separación).
            gap = my_mins_to_aep - lead_mins_to_aep

            if gap < SEPARACION_PELIGRO:
                # Como antes: tratar de frenar 20 kts por debajo del líder. Si no alcanza, turnaround.
                nueva_vel = min(vmax, speed_prev[leader_id] - 20.0)
                if nueva_vel < vmin:
                    av.estado = "turnaround"
                    av.velocidad_kts = VEL_TURNAROUND
                    self.mover_a_turnaround(aid)
                    self.recien_turnaround.add(aid)  # para no moverlo en este mismo step
                else:
                    av.velocidad_kts = max(vmin, nueva_vel)
                continue  # ya resolvimos el caso de peligro; saltamos metering

            # --- 2) Política A (metering anticipado) en zona segura ---
            # Predicción de gap si YO me mantuviera a vmax (no hay peligro, pero puede haber "compresión").
            my_mins_to_aep_vmax = mins_a_aep(dist_prev[aid], vmax)
            gap_pesimista = my_mins_to_aep_vmax - lead_mins_to_aep
            pred_gap = gap_pesimista  # ya lo calculamos con my_mins_to_aep_vmax

            # Target deseado según distancia (más grande lejos), con un pequeño buffer
            g_target = g_objetivo(dist_prev[aid])

            if pred_gap < g_target + BUFFER_ANTICIPACION:
                # Queremos llegar a t_target = ETA_líder + g_target
                t_target = lead_mins_to_aep + g_target
                # Velocidad requerida (nm/min -> kts) respetando límites físicos de la banda
                v_req = (dist_prev[aid] / t_target) * 60.0 if t_target > 0 else vmax
                v_req = min(max(v_req, vmin), vmax)
                av.velocidad_kts = v_req
            else:
                # Ya hay suficiente aire -> mantener vmax (política "no interferir")
                av.velocidad_kts = vmax

        # Intento de reingreso global desde turnaround (tu lógica original)
        activos_order = list(self.activos)
        self.intentar_reingreso(activos_order)
