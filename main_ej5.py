# -*- coding: utf-8 -*-
"""
Simulador de aproximaciones con dos carriles y listas separadas.
Cumple con:
- Lista de vuelos activos (approach)
- Lista de vuelos en turnaround (segundo carril)
- Lista de vuelos inactivos (diverted o landed)
- Cada avión mantiene puntero a su líder (lista simplemente enlazada por mins_to_aep)
- IDs únicos e inmutables para mapear aviones
- Reingreso desde turnaround buscando huecos GLOBALES y ajustando velocidad objetivo

Diseño: un TraficoAviones orquesta aparicions, control y movimiento por pasos.
Unidades: distancia en nm, velocidades en kts, tiempo en minutos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import animation
import random

# -----------------------
# Constantes y utilidades
# -----------------------
MINUTE = 1.0
SEPARACION_MINIMA = 5.0  # separación deseada a ambos lados (min)
SEPARACION_PELIGRO = 4.0 # debajo de esto, hay conflicto (actuar fuerte)
VEL_TURNAROUND = 200.0   # kts alejándose
MAX_DIVERTED_DISTANCE = 100.0      # si se aleja más de 100 nm -> diverted
DAY_START = 0
DAY_END = 1080 

# Bandas (dist_lo, dist_hi, vmin, vmax)
VELOCIDADES = [
    (100.0, float('inf'), 300.0, 500.0),
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

def knots_to_nm_per_min(k: float) -> float:
    ''' convierte de nudos a nm/min (ambas son unidades de velocidad)'''
    return k / 60.0

def velocidad_por_distancia(d_nm: float) -> Tuple[float, float]:
    for dist_low, dist_high, vmin, vmax in VELOCIDADES:
        if dist_low <= d_nm < dist_high:
            return vmin, vmax # devuelve las velocidades para esa banda
    return VELOCIDADES[0][2], VELOCIDADES[0][3] # si te pasas de los 100nm


def mins_a_aep(dist_nm: float, speed_kts: float) -> float:
    ''' cuantos minutos te faltan para llegar a aep si vas a speed_kts constante '''
    t = 0.0
    d = dist_nm
    v = speed_kts
    while d > 0:
        for dist_low, dist_high, vmin, vmax in VELOCIDADES:
            if dist_low <= d < dist_high: # estoy en esta banda
                # distancia restante en la banda actual
                dist_banda = min(d, d - dist_low)
                if dist_banda <= 0:
                    # Si no hay distancia para recorrer, salta a la siguiente banda
                    d -= 1e-6
                    continue
                # tiempo para recorrer esa distancia a velocidad actual
                t_banda = dist_banda / knots_to_nm_per_min(v)
                t += t_banda
                d -= dist_banda
                # al pasar de banda, actualizo velocidad a vmax de la banda siguiente
                v = vmin # vmin de mi banda actual es vmax de la siguiente
                break
    return t



# -----------------
# objeto Avión para simulación
# -----------------
@dataclass
class Avion:
    id: int
    aparicion_min: int # minuto en el que aparece
    distancia_nm: float = 100.0
    velocidad_kts: float = 300.0
    estado: str = "approach"  # approach | turnaround | interrupted | diverted | landed
    leader_id: Optional[int] = None  # puntero a su líder en el carril approach (como si fuese una lista simplemente enlazada)
    # --- campos extra para estadística ---
    aterrizaje_min: Optional[int] = None  # minuto en que aterrizó (si aplica)

    def limites_velocidad(self) -> Tuple[float, float]:
        return velocidad_por_distancia(self.distancia_nm)

    def tiempo_a_aep(self, speed: Optional[float] = None) -> float:
        v = self.velocidad_kts if speed is None else speed
        return mins_a_aep(self.distancia_nm, v)

class TraficoAviones:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed) # genera num aleatorios para apariciones
        self.next_id = 1 # próximo id a asignar (va incrementándose cada vez que aparece un avión)
        self.planes: Dict[int, Avion] = {} # mapeo id -> Avion para acceder más rápido (sin tener que recorrer la lista)
        self.activos: List[int] = []      # ids en estado approach (ordenados por mins_to_aep ascendente)
        self.turnaround: List[int] = []    # ids en estado turnaround
        self.inactivos: List[int] = []     # ids en estado diverted | landed
        self.recien_turnaround: Set[int] = set()  # ids de aviones que cambiaron a turnaround este paso (esto es para que no retrocedan en el mismo paso que cambian de estado)
        self.interrupted: List[int] = []   # ids de aviones interrumpidos
        self.recien_interrupted: Set[int] = set()  # ids de aviones que recién pasaron a interrupted
        self.current_min: int = DAY_START


    def aparicion(self, minuto: int) -> Avion:
        ''' crear un avion'''
        aid = self.next_id # id que le asigno al avion que apareció
        self.next_id += 1 # actualizo para el próximo avión que aparezca
        av = Avion(id=aid, aparicion_min=minuto, distancia_nm=100.0, velocidad_kts=300.0, estado="approach")
        self.planes[aid] = av
        self.activos.append(aid) 
        return av

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
        self.recien_interrupted.clear()

        # Decidir en carril approach
        for aid in list(self.activos):
            av = self.planes[aid]
            vmin, vmax = velocidad_por_distancia(dist_prev[aid])

            # Chequeo de interrupción (go-around por viento)  (EJERCICIO 5)
            if dist_prev[aid] <= 20.0:  # solo en la fase final (<20 nm)
                if self.rng.random() < 0.1:  # 1/10 de probabilidad
                    av.estado = "interrupted"
                    av.velocidad_kts = VEL_TURNAROUND
                    self.mover_a_interrupted(aid)
                    continue  # salto a siguiente avión, no sigo con la lógica normal

            leader_id = av.leader_id
            if leader_id is None:
                av.velocidad_kts = vmax
                continue

            my_mins_to_aep = mins_a_aep(dist_prev[aid], vmax)
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

    def mover_a_turnaround(self, aid: int) -> None:
        ''' mueve un avión del carril activo al carril turnaround '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid not in self.turnaround:
            self.turnaround.append(aid)
        self.planes[aid].leader_id = None # ya no tiene líder en el carril approach

    def mover_a_interrupted(self, aid: int) -> None:
        ''' mueve un avión del carril activo al carril interrupted '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid not in self.interrupted:
            self.interrupted.append(aid)
            self.recien_interrupted.add(aid)
        self.planes[aid].leader_id = None # ya no tiene líder en el carril approach

    def mover_a_activos(self, aid: int) -> None:
        ''' mueve un avión del carril turnaround al carril activo cuando reingresa '''
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid not in self.activos:
            self.activos.append(aid)
    
    def mover_interrupted_a_activos(self, aid: int) -> None:
        if aid in self.interrupted:
            self.interrupted.remove(aid)
        if aid not in self.activos:
            self.activos.append(aid)


    def mover_a_inactivos(self, aid: int) -> None:
        ''' mueve un avión del carril activo o turnaround a inactivos (landed o diverted) '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid in self.turnaround:
            self.turnaround.remove(aid)
        if aid in self.interrupted:
            self.interrupted.remove(aid)
        if aid not in self.inactivos:
            self.inactivos.append(aid)
        self.planes[aid].leader_id = None

    # Reingreso desde turnaround: busca huecos globales y ajusta velocidad
    def intentar_reingreso(self, activos_order: List[int]) -> None:
        """
        Intenta reinsertar aviones que están en turnaround (por congestión) o en interrupted (por go-around).
        La lógica es la misma para ambos:
        - Buscar huecos de al menos 10 minutos en la cola de approach (5 min al avión de adelante y 5 min al de atrás).
        - Ajustar velocidad para caer en ese hueco.
        - Reinsertar en approach si se encuentra un rango factible de velocidades.
        Condiciones adicionales:
        - Aviones en "interrupted" NO pueden reinsertarse si ya están a ≤ 5 nm de AEP.
        """
        if not activos_order: 
            # Caso borde: si no hay ningún avión en approach,
            # todos los que estén en turnaround o interrupted pueden volver directo a approach a vmax.
            for aid in list(self.turnaround) + list(self.interrupted):
                av = self.planes[aid]
                vmin, vmax = av.limites_velocidad()
                av.velocidad_kts = vmax  # aceleran a su velocidad máxima permitida
                av.estado = "approach"   # cambian estado a approach
                if aid in self.turnaround:
                    self.mover_a_activos(aid)  # limpia de turnaround y agrega a approach
                elif aid in self.interrupted:
                    self.mover_interrupted_a_activos(aid)  # limpia de interrupted y agrega a approach
            return

        # Lista de (id, tiempo estimado a AEP) de todos los aviones en approach
        # Esto permite identificar huecos temporales disponibles en la secuencia de aterrizajes
        activos_mins_to_aep = [(aid, self.planes[aid].tiempo_a_aep()) for aid in activos_order]
        activos_mins_to_aep.sort(key=lambda x: x[1])  # ordenar por tiempo de llegada

        # Recorro todos los aviones que podrían reinsertarse (turnaround + interrupted)
        for aid in list(self.turnaround) + list(self.interrupted):
            av = self.planes[aid]
            d = av.distancia_nm  # distancia actual al AEP
            vmin, vmax = av.limites_velocidad()  # límites de velocidad según la banda de distancia
            mins_to_aep_fast = mins_a_aep(d, vmax)  # tiempo de llegada si fuese a vmax
            mins_to_aep_slow = mins_a_aep(d, vmin)  # tiempo de llegada si fuese a vmin

            reinsertado = False

            # Restricción especial para interrupted:
            # No se puede reinsertar si ya está demasiado cerca del aeropuerto (<5 nm).
            if aid in self.interrupted and d <= 5.0:
                continue

            # 🔹 Caso principal: buscar huecos entre pares de aviones consecutivos en approach
            for (a1, mins_to_aep1), (a2, mins_to_aep2) in zip(activos_mins_to_aep, activos_mins_to_aep[1:]):
                t_low = mins_to_aep1 + SEPARACION_MINIMA   # límite inferior del hueco (5 min después del avión de adelante)
                t_high = mins_to_aep2 - SEPARACION_MINIMA  # límite superior del hueco (5 min antes del avión de atrás)

                if t_high < t_low:
                    # Si no hay suficiente espacio entre estos dos aviones, no es un hueco válido
                    continue

                # Rango factible para este avión según velocidades y hueco disponible
                a = max(t_low, mins_to_aep_fast)  # no puedo llegar antes de t_low ni más rápido que vmax
                b = min(t_high, mins_to_aep_slow) # no puedo llegar después de t_high ni más lento que vmin

                if a <= b:
                    # Si el rango [a, b] es válido, me posiciono en el medio del hueco
                    t_target = (a + b) / 2.0
                    v_target = (d / t_target) * 60.0  # velocidad requerida en kts para llegar en t_target
                    v_target = min(max(v_target, vmin), vmax)  # ajustar para no salir de límites físicos

                    my_mins_to_aep = mins_a_aep(d, v_target)  # recalculo con la velocidad corregida

                    if t_low <= my_mins_to_aep <= t_high:
                        # El avión puede reinsertarse en ese hueco
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        if aid in self.turnaround:
                            self.mover_a_activos(aid)
                        elif aid in self.interrupted:
                            self.mover_interrupted_a_activos(aid)
                        reinsertado = True
                        break  # ya encontró hueco, no sigue buscando

            # Caso borde: probar antes del primer avión de la cola
            if not reinsertado:
                first_mins_to_aep = activos_mins_to_aep[0][1]
                t_high = first_mins_to_aep - SEPARACION_MINIMA  # tengo que llegar al menos 5 min antes del primero
                a = mins_to_aep_fast
                b = min(t_high, mins_to_aep_slow)
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_mins_to_aep = mins_a_aep(d, v_target)
                    if my_mins_to_aep <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        if aid in self.turnaround:
                            self.mover_a_activos(aid)
                        elif aid in self.interrupted:
                            self.mover_interrupted_a_activos(aid)
                        reinsertado = True

            # Caso borde: probar después del último avión de la cola
            if not reinsertado:
                last_mins_to_aep = activos_mins_to_aep[-1][1]
                t_low = last_mins_to_aep + SEPARACION_MINIMA  # tengo que llegar al menos 5 min después del último
                a = max(t_low, mins_to_aep_fast)
                b = mins_to_aep_slow
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = d / t_target * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_mins_to_aep = mins_a_aep(d, v_target)
                    if my_mins_to_aep >= t_low:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        if aid in self.turnaround:
                            self.mover_a_activos(aid)
                        elif aid in self.interrupted:
                            self.mover_interrupted_a_activos(aid)
                        reinsertado = True

            # Si no encontró hueco válido en ningún caso → sigue en turnaround o interrupted


    def mover_paso(self) -> None:
        # approach
        for aid in list(self.activos):
            av = self.planes[aid]
            avance_nm = knots_to_nm_per_min(av.velocidad_kts) * MINUTE
            av.distancia_nm = max(0.0, av.distancia_nm - avance_nm)
            if av.distancia_nm <= 0.0: # llegó a aep
                av.distancia_nm = 0.0
                av.velocidad_kts = 0.0
                av.estado = "landed"
                av.aterrizaje_min = self.current_min
                self.mover_a_inactivos(aid)
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

        # interrupted
        for aid in list(self.interrupted):
            av = self.planes[aid]
            if aid in self.recien_interrupted:
                # no mover en el mismo paso del cambio a interrupted
                continue
            retro_nm = knots_to_nm_per_min(VEL_TURNAROUND) * MINUTE
            av.distancia_nm += retro_nm
            if av.distancia_nm >= MAX_DIVERTED_DISTANCE:
                av.estado = "diverted"
                self.mover_a_inactivos(aid)

        # recalcular orden y líderes para próximo paso
        self.ordenar_activos()

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

    def aviones_landed(self) -> List[Avion]:
        return [av for av in self.planes.values() if av.aterrizaje_min is not None]
    # filtra solo los aviones que realmente aterrizaron 


#################
#  funciones para visualizacion
#################
def snapshot_frame(ctrl: TraficoAviones) -> Dict[str, np.ndarray]:
    # Mapea cada avión a una coordenada (x, y) y color según estado.
    xs = []
    ys = []
    cs = []
    # Carriles: approach en y=1, turnaround en y=0
    for aid in ctrl.activos:
        av = ctrl.planes[aid]
        xs.append(av.distancia_nm)
        ys.append(1.0)
        cs.append('tab:blue')
    for aid in ctrl.turnaround:
        av = ctrl.planes[aid]
        xs.append(av.distancia_nm)
        ys.append(0.0)
        cs.append('tab:red')

    for aid in ctrl.interrupted:
        av = ctrl.planes[aid]
        xs.append(av.distancia_nm)
        ys.append(0.5)  # carril intermedio para diferenciarlos (entre 0.0 y 1.0)
        cs.append('tab:orange')  # color naranja para los interrupted

    # Inactivos: apilamos con pequeños offsets en Y para ver múltiples
    landed_idx = 0
    diverted_idx = 0
    y_step = 0.03  # separación visual mínima
    # Landed: x=0 fijo, y parte de 1.0 hacia abajo en escalones
    # Diverted: x=110 fijo, y parte de -0.5 hacia arriba en escalones
    for aid in ctrl.inactivos:
        av = ctrl.planes[aid]
        if av.estado == 'landed':
            xs.append(0.0)
            ys.append(1.0 - landed_idx * y_step)
            cs.append('tab:green')
            landed_idx += 1
        elif av.estado == 'diverted':
            xs.append(110.0)
            ys.append(-0.5 + diverted_idx * y_step)
            cs.append('gray')
            diverted_idx += 1
    return {
        'x': np.array(xs, dtype=float),
        'y': np.array(ys, dtype=float),
        'c': np.array(cs, dtype=object),
    }

def save_gif_frames(frames: List[Dict[str, np.ndarray]], out_path: str = "simulaciones/sim.gif", fps: int = 10, label_text: Optional[str] = None):
    # Configuración de figura
    x_max = 120.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, x_max)
    ax.set_ylim(-1.0, 2.0)
    ax.invert_xaxis()
    ax.set_xlabel("Distancia a AEP (nm)")
    ax.set_ylabel("Carril / estado")
    # Etiqueta arriba a la izquierda en coordenadas de figura para no tapar puntos
    if label_text:
        fig.text(0.015, 0.985, label_text, ha='left', va='top')

    scat = ax.scatter([], [])
    scat.set_offsets(np.empty((0, 2)))

    # Leyenda fija
    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None', color='tab:blue',   label='approach'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:red',    label='turnaround'),
        Line2D([0],[0], marker='o', linestyle='None', color='gray',       label='diverted'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:green',  label='landed'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:orange', label='interrupted'),

    ]
    ax.legend(handles=legend_handles, loc='upper right')
    # Línea de referencia en 100 nm (fina y punteada)
    ax.axvline(100.0, color='k', linestyle=':', linewidth=0.8, alpha=0.7)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_color([])
        return scat,

    def update(i):
        f = frames[i]
        pts = np.column_stack((f['x'], f['y']))
        scat.set_offsets(pts)
        scat.set_color(f['c'])
        ax.set_title(f"minuto {i}")
        return scat,

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(frames), interval=1000//fps, blit=True
    ) 
    try:
        anim.save(out_path, writer='pillow', fps=fps)
        print(f"GIF guardado en {out_path}")
    except Exception as e:
        print(f"No se pudo guardar el GIF: {e}")

if __name__ == "__main__":
    ######
    #! Simulación
    ######
    # a mayor lambda, menos estricto el umbral que tiene que cumplir la proba random que se le asigna a cada minuto, entonces: más probabilidad de que aparezca un avión en cada minuto
    lamb = 0.1

    trafico_sim = TraficoAviones(seed=42)
    apariciones = set(trafico_sim.bernoulli_aparicion(lamb))

    frames: List[Dict[str, np.ndarray]] = []
    for t in range(DAY_START, DAY_END):
        trafico_sim.step(t, aparicion=(t in apariciones))
        # capturo frame cada minuto para la visualizacion
        frames.append(snapshot_frame(trafico_sim))

    #! GIF
    # Aumento el fps para que avance más rápido
    l = str(lamb).replace(".", "")
    gif_name = f"sim_lambda{l}" + ".gif"
    save_gif_frames(frames, out_path=f"simulaciones/{gif_name}", fps=5, label_text=f"λ = {lamb}")

