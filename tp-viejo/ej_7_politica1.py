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

Extensión (Política A - "metering" anticipado):
- En zona segura (gap >= SEPARACION_PELIGRO), cada seguidor ajusta su velocidad
  para apuntar a llegar en t_target = t_leader + g_target(d), donde g_target(d)
  aumenta un poco en tramos lejanos (cola "río arriba") para evitar compresiones.
- Si yendo a vmax el gap ya es cómodo, se mantiene vmax (no se frena de más).
- Siempre se respetan vmin/vmax por banda y se preserva el fallback de seguridad.
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

# Parámetros de la Política A (metering anticipado)
OBJ_SEP_BASE = 5.5         # target "cómodo" (>5). Podés barrerlo para ver tradeoff atraso↔desvíos
BUFFER_ANTICIPACION = 0.5  # anticipo para empezar a ajustar un poco antes

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
    ''' Devuelve (vmin, vmax) de la banda correspondiente a la distancia d_nm. '''
    for dist_low, dist_high, vmin, vmax in VELOCIDADES:
        if dist_low <= d_nm < dist_high:
            return vmin, vmax
    # si te pasas de los 100nm (raro por cómo seteamos apariciones), usar la banda más lejana
    return VELOCIDADES[0][2], VELOCIDADES[0][3]

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

def g_objetivo(d_nm: float) -> float:
    """
    Target de separación deseada en minutos según distancia.
    - Más lejos pedimos un poquito más para evitar compresiones aguas abajo.
    - En tramo final, mantenemos 5 min (separación mínima operacional).
    """
    if d_nm >= 50.0:
        return OBJ_SEP_BASE + 1.0   # p.ej. 7 si base=6
    if d_nm >= 15.0:
        return OBJ_SEP_BASE         # p.ej. 6
    return 4.0                      # tramo final 5 min

# -----------------
# objeto Avión para simulación
# -----------------
@dataclass
class Avion:
    id: int
    aparicion_min: int # minuto en el que aparece
    distancia_nm: float = 100.0
    velocidad_kts: float = 300.0
    estado: str = "approach"  # approach | turnaround | diverted | landed
    leader_id: Optional[int] = None  # puntero a su líder en el carril approach

    def limites_velocidad(self) -> Tuple[float, float]:
        return velocidad_por_distancia(self.distancia_nm)

    def tiempo_a_aep(self, speed: Optional[float] = None) -> float:
        v = self.velocidad_kts if speed is None else speed
        return mins_a_aep(self.distancia_nm, v)

class TraficoAviones:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed) # genera num aleatorios para apariciones
        self.next_id = 1 # próximo id a asignar
        self.planes: Dict[int, Avion] = {}
        self.activos: List[int] = []      # ids en approach (ordenados por ETA ascendente)
        self.turnaround: List[int] = []   # ids en turnaround
        self.inactivos: List[int] = []    # ids en diverted | landed
        # Set de ids que acaban de pasar a turnaround para no moverlos en el mismo paso
        self.recien_turnaround: Set[int] = set()

    def aparicion(self, minuto: int) -> Avion:
        ''' crear un avion'''
        aid = self.next_id
        self.next_id += 1
        av = Avion(id=aid, aparicion_min=minuto, distancia_nm=100.0, velocidad_kts=300.0, estado="approach")
        self.planes[aid] = av
        self.activos.append(aid)
        return av

    def ordenar_activos(self, current_speeds: Optional[Dict[int, float]] = None):
        '''
        current_speeds es una "foto" de las velocidades actuales (id -> vel_kts).
        Se usa para ordenar por ETA consistente dentro del paso.
        '''
        if not self.activos:
            return
        def tiempo_estimado(aid: int) -> float:
            av = self.planes[aid]
            v = av.velocidad_kts if current_speeds is None else current_speeds[aid]
            return av.tiempo_a_aep(v)
        self.activos.sort(key=tiempo_estimado)
        # actualizar punteros a líder (anterior en ETA)
        for i, aid in enumerate(self.activos):
            av = self.planes[aid]
            av.leader_id = self.activos[i-1] if i > 0 else None

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

    def mover_a_turnaround(self, aid: int) -> None:
        ''' mueve un avión del carril activo al carril turnaround '''
        if aid in self.activos:
            self.activos.remove(aid)
        if aid not in self.turnaround:
            self.turnaround.append(aid)
        self.planes[aid].leader_id = None

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

    # Reingreso desde turnaround: busca huecos globales y ajusta velocidad
    def intentar_reingreso(self, activos_order: List[int]) -> None:
        if not activos_order:
            # Sin nadie en approach -> todos reingresan directo a vmax
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
            mins_to_aep_fast = mins_a_aep(d, vmax)
            mins_to_aep_slow = mins_a_aep(d, vmin)

            reinsertado = False
            # Buscar gap entre pares consecutivos
            for (a1, t1), (a2, t2) in zip(activos_mins_to_aep, activos_mins_to_aep[1:]):
                t_low  = t1 + SEPARACION_MINIMA
                t_high = t2 - SEPARACION_MINIMA
                if t_high < t_low:
                    continue

                a = max(t_low, mins_to_aep_fast)
                b = min(t_high, mins_to_aep_slow)
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = (d / t_target) * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_eta = mins_a_aep(d, v_target)
                    if t_low <= my_eta <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        reinsertado = True
                        break

            # Antes del primero
            if not reinsertado:
                first_eta = activos_mins_to_aep[0][1]
                t_high = first_eta - SEPARACION_MINIMA
                a = mins_to_aep_fast
                b = min(t_high, mins_to_aep_slow)
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = (d / t_target) * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_eta = mins_a_aep(d, v_target)
                    if my_eta <= t_high:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        reinsertado = True

            # Después del último
            if not reinsertado:
                last_eta = activos_mins_to_aep[-1][1]
                t_low = last_eta + SEPARACION_MINIMA
                a = max(t_low, mins_to_aep_fast)
                b = mins_to_aep_slow
                if a <= b:
                    t_target = (a + b) / 2.0
                    v_target = (d / t_target) * 60.0
                    v_target = min(max(v_target, vmin), vmax)
                    my_eta = mins_a_aep(d, v_target)
                    if my_eta >= t_low:
                        av.velocidad_kts = v_target
                        av.estado = "approach"
                        self.mover_a_activos(aid)
                        reinsertado = True
            # Si no reinsertó: sigue en turnaround

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
        if aparicion:
            self.aparicion(minuto)
        self.control_paso()
        self.mover_paso()


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
    # Inactivos: apilamos con pequeños offsets en Y para ver múltiples
    landed_idx = 0
    diverted_idx = 0
    y_step = 0.03
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
    if label_text:
        fig.text(0.015, 0.985, label_text, ha='left', va='top')

    scat = ax.scatter([], [])
    scat.set_offsets(np.empty((0, 2)))

    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None', color='tab:blue',   label='approach'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:red',    label='turnaround'),
        Line2D([0],[0], marker='o', linestyle='None', color='gray',       label='diverted'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:green',  label='landed'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')
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
    lamb = 0.1
    trafico_sim = TraficoAviones(seed=42)
    apariciones = set(trafico_sim.bernoulli_aparicion(lamb))

    frames: List[Dict[str, np.ndarray]] = []
    for t in range(DAY_START, DAY_END):
        trafico_sim.step(t, aparicion=(t in apariciones))
        frames.append(snapshot_frame(trafico_sim))

    l = str(lamb).replace(".", "")
    gif_name = f"sim_lambda{l}" + ".gif"
    save_gif_frames(frames, out_path=f"simulaciones/{gif_name}", fps=5, label_text=f"λ = {lamb}")
