
# TraficoAEPPolitica.py

from __future__ import annotations
from typing import Dict, List, Optional, Set

from TraficoAEP import TraficoAviones
from Constants import SEPARACION_PELIGRO, VEL_TURNAROUND, BUFFER_ANTICIPACION
from Helpers import velocidad_por_distancia, mins_a_aep, g_objetivo


####################
# Política 1: Metering anticipado con fallback de seguridad
####################
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

####################
# Política 2a: reingreso con prioridad por riesgo de desvío
####################

class TraficoAvionesPolitica2a(TraficoAviones):

    # Reingreso desde turnaround: busca huecos globales y ajusta velocidad
    def intentar_reingreso(self, activos_order: List[int]) -> None:
        #!politica 2a: reingreso primero a los que estan en riesgo de desviarse (en 'turnaround', mas lejos de AEP)
        # ordeno los turnaround por distancia a AEP descendente (los mas lejos primero)
        turnaround_sorted = sorted(self.turnaround, key=lambda aid: self.planes[aid].distancia_nm, reverse=True)
        if not activos_order: # si no hay ningun avion en el carril de activos --> reingresan todos a vmax (es como darlos vuelta de dirección y pasarlos de carril)
            for aid in turnaround_sorted:
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                self.mover_a_activos(aid)
            return

        activos_mins_to_aep = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_mins_to_aep.sort(key=lambda x: x[1])

        for aid in turnaround_sorted:
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
                    v_target = min(max(v_target, vmin), vmax) # el max descarta vt1, el min descarta vt3

                    my_mins_to_aep = mins_a_aep(d, v_target)
                    
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


####################
# Política 2b: reingreso con prioridad FIFO
####################

class TraficoAvionesPolitica2b(TraficoAviones):

    # Reingreso desde turnaround: busca huecos globales y ajusta velocidad
    def intentar_reingreso(self, activos_order: List[int]) -> None:
        #!politica 2b: reingreso con orden FIFO de turnaround (el que más tiempo lleva en turnaround reingresa primero)
        turnaround_sorted = self.turnaround.copy() # como self.turnaround se va actualizando con un append, esta en orden FIFO
        
        if not activos_order: # si no hay ningun avion en el carril de activos --> reingresan todos a vmax (es como darlos vuelta de dirección y pasarlos de carril)
            for aid in turnaround_sorted:
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax
                av.estado = "approach"
                self.mover_a_activos(aid)
            return

        activos_mins_to_aep = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_mins_to_aep.sort(key=lambda x: x[1])

        for aid in turnaround_sorted:
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
                    v_target = min(max(v_target, vmin), vmax) # el max descarta vt1, el min descarta vt3

                    my_mins_to_aep = mins_a_aep(d, v_target)
                    
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