from __future__ import annotations
import matplotlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math, random
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.animation as animation

#! Conversion de Unidades
def nm_to_km(nm: float) -> float: return nm * 1.852
def knots_to_kmh(k: float) -> float: return k * 1.852
def knots_to_nm_per_min(k: float) -> float: return k / 60.0  # nm/min

#! Parametros globales del problema
MINUTE = 1.0
separacion_minima = 4.0  # min
separacion_target = 5.0  # min (buffer)
velocidad_reversa = 200.0  # kts (hacia atrás)
apertura_aep_min = 6*60
cierre_aep_min = 24*60  # medianoche (medimos desde 00:00)
minutos_del_dia = cierre_aep_min - apertura_aep_min  # 1080 min
x_max = 120.0

velocidades = [
    # (dist_min_inclusive, dist_max_exclusive, vmin, vmax)  en nmi y kts
    (100.0, float('inf'), 300.0, 500.0),  # >100 nm
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

#! Funciones Utiles a lo largo del problema
def velocidad_por_distancia(d_nm: float) -> Tuple[float, float]:
    for lo, hi, vmin, vmax in velocidades:
        if lo <= d_nm < hi:
            return vmin, vmax
    # si está más allá de 100 nm:
    return velocidades[0][2], velocidades[0][3]

def eta_min(dist_nm: float, speed_kts: float) -> float:
    """Tiempo a pista (min) asumiendo velocidad constante."""
    return float('inf') if speed_kts <= 0 else dist_nm / (speed_kts / 60.0)

def gap_minutos(self_dist: float, self_speed: float, lead_dist: float, lead_speed: float) -> float:
    """ETA(seguidor) - ETA(líder) en minutos, con velocidades actuales."""
    return eta_min(self_dist, self_speed) - eta_min(lead_dist, lead_speed)

def encontrar_lider(self: "Avion", cohort: Dict[int, "Avion"]):
    #*Devuelve el avion por delante mas cercano que no este en diverted, landed o turnaround *#
    candidatos = [a for a in cohort.values() if a.id != self.id and a.distancia_a_aep < self.distancia_a_aep and a.status in("approach", "delayed")]
    if not candidatos:
        return None
    
    return max(candidatos, key=lambda a: a.distancia_a_aep)


#! Clase Avion y sus metodos 
@dataclass
class Avion:
    id: int
    momento_aparicion: float  # minuto en que aparece a 100 nm
    distancia_a_aep: float = 100.0
    velocidad: float = 300.0
    status: str = "approach"  # approach | delayed | turnaround | diverted | landed


    def velocidad_permitida(self) -> Tuple[float, float]:
        return velocidad_por_distancia(self.distancia_a_aep)
    
 
    def step(self, *, cohort: Dict[int, "Avion"],
                     dt_min: float = 1.0,
                     gap_target_min: float = separacion_target,
                     gap_minimo_min: float = separacion_minima,
                     gap_reingreso_min: float = 10.0,)-> None:
        
        #* Aplica la regla por minuto, SIN métricas ni historial (Ej. 1).
        #*- Si no hay líder a < 4 min => vmax de su banda (approach).
        #*- Si hay líder con gap < 4 min => baja 20 kts vs líder hasta lograr >= 5 min (delayed).
        #*- Si para lograr gap necesita < vmin => sale a 'turnaround' (200 kts hacia atrás).
        #*- En 'turnaround' busca gap >= 10 min; si pasa >100 nm => 'diverted'.
        

        #*Set del avion en el minuto i, donde si aterrizo o se fue a Montevideo no me importa mas y sino chequeo si tiene algun avion por delante para evaluar potenciales cambios en su andar
        if self.status in ("landed", "diverted"):
            return
        
        leader = encontrar_lider(self, cohort)

        vmin, vmax = velocidad_por_distancia(self.distancia_a_aep)

        #*En el caso que el avion esta en turnaround 
        if self.status == "turnaround":
            if leader is None:
                self.status = "approach"
                self.velocidad = vmax

            else:
                gap = gap_minutos(self.distancia_a_aep, vmax, leader.distancia_a_aep, leader.velocidad)
                if gap >= gap_reingreso_min:
                    self.status = "approach"
                    self.velocidad = vmax
                else:
                    self.velocidad = velocidad_reversa
                    self.distancia_a_aep += knots_to_nm_per_min(self.velocidad) * dt_min
                    if self.distancia_a_aep >= 100.0:
                        self.status = "diverted"
                    return
                


        else:
            velDes = vmax
            if leader is None:
                self.velocidad = velDes
                self.status = "approach"
            else:
                #* Quiero calcular el gap si yo voy a la velocidad maxima que me permiten y el de adelante a la velocidad que vaya
                gap = gap_minutos(self.distancia_a_aep, velDes, leader.distancia_a_aep, leader.velocidad)
                if gap < gap_minimo_min: #!Estoy a menos de 4min
                    nuevaVel = min(velDes, leader.velocidad - 20.0)
                    if nuevaVel < vmin: #!me fui por debajo de los limites
                        self.status = "turnaround"
                        self.velocidad = velocidad_reversa

                        self.distancia_a_aep += knots_to_nm_per_min(self.velocidad) * dt_min #!Hago de cuenta como que avance un minuto para ver si me fui de las 100 mn
                        if self.distancia_a_aep > 100.0:
                            self.status = "diverted"
                        return
                    else:
                        self.velocidad = nuevaVel
                        self.status = "delayed"

                elif gap >= gap_target_min: #!Consegui un gap de 5min o mas
                    self.status = "approach"
                    self.velocidad = velDes
                
                else: #!Estoy entre 4 y 5 minutos -> sigo todavia puedo ir a la velocidad que quiera
                    self.velocidad = velDes
                    self.status = "delayed"
        
        avance_mn = knots_to_nm_per_min(self.velocidad) *dt_min
        self.distancia_a_aep = max(0.0, self.distancia_a_aep - avance_mn)

        if self.status != "turnaround" and self.distancia_a_aep <= 0.0:
            self.status = "landed"
            self.velocidad = 0.0
            return



#!Defino la funcion que representa las llegadas Bernoulli por minuto
def tiempos_de_spawn(lam_per_min, t0, t1, seed=42):
    
    rng = random.Random(seed)
    tiempos = []
    for i in range (t0, t1):
        u = rng.random()
        if u < lam_per_min:
            tiempos.append(i)
    return tiempos

#!Defino otra funcion que define los estados y sus colores asociados que se van a usar posteriormente en la visualizacion
def color_estados(estado):
    return {
        "approach": "tab:blue",
        "delayed":    "tab:orange",
        "turnaround": "tab:red",
        "diverted":   "tab:green",
        "landed": "tab:gray"
    }.get(estado, "diverted")



#!Ahora la simulacion 

def simular(lam, t_inicio, t_final, seed = 42):
    
    rng = random.Random(seed)

    #*Importante: Para que la simulacion sea consistente con la funcion step que defini antes es importante que los id's de aviones 
    #*vayan en orden decreciente asi el avion con mayor id es el que mas cerca de AEP se encuentre

    prox_avion_id = 0

    vuelos: List[Avion] = []
    cohort: Dict[int, Avion] = {}  #*Cohort es un metodo que sirve para poder trackear el avion con solamente el id y no tener que recorrer toda la lista (complejidad O(N))

    spawns = set(tiempos_de_spawn(lam, t_inicio, t_final, seed))

    #*Lo siguiente es solo para la visualizacion
    lanes: Dict[int, int] = {}
    lane_counter = 0
    lane_step = 1.2

    frames = []


    for t in range(t_inicio, t_final):

        if t in spawns:
            prox_avion_id -= 1
            av = Avion(prox_avion_id, t, 100.0, 300.0, "approach")
            vuelos.append(av)
            cohort[av.id] = av
            lanes[av.id] = lane_counter
            lane_counter += 1

        
        for f in sorted(vuelos, key=lambda a: a.id):  #?Esto todavia no entendi para que sirve pero es IMPORTANTE
            f.step(cohort=cohort, dt_min=1.0)


        if t >= 0: #*Esto solo para empezar a capturar la simulacion a partir del t0 en caso de que hayas hecho un warm up desde numeros anteriores
            xs, ys, cs = [], [], []
            for v in vuelos:
                if v.status in ("diverted", "landed"):
                    continue 
                xs.append(max(0.0, min(x_max, v.distancia_a_aep)))
                ys.append(lanes[v.id] * lane_step) #*Valores irreales para calcular la posicion vertical del avion
                cs.append(color_estados(v.status))
            frames.append((xs, ys, cs))

    return frames, lanes



#!Visualizacion (por ahora todo chat)
def save_gif_frames(frames, lanes, out_path="sim_ej1.gif", fps=10):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, x_max)       
    ax.set_ylim(-1, max(2, len(lanes)) * 1.2)
    ax.invert_xaxis()
    ax.set_xlabel("Distancia a AEP (nm)")
    ax.set_ylabel("Pista visual por avión")

 
    vline_100 = ax.axvline(
        100, linestyle="--", color="k", alpha=0.6, linewidth=1.5, zorder=1
    )

    scat = ax.scatter([], [])
    scat.set_offsets(np.empty((0, 2)))

    # Leyenda
    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None', color='tab:blue',   label='approach'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:orange', label='delayed'),
        Line2D([0],[0], marker='o', linestyle='None', color='tab:red',    label='turnaround'),
        Line2D([0],[0], marker='o', linestyle='None', color='gray',       label='diverted'),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    def init():
        scat.set_offsets(np.empty((0, 2)))  
        scat.set_color([])                  
        return (scat, vline_100)

    def update(i):
        xs, ys, cs = frames[i]
        if xs:  
            offs = np.column_stack([xs, ys])  
            scat.set_offsets(offs)
            scat.set_color(cs)                 
        else:  
            scat.set_offsets(np.empty((0, 2)))
            scat.set_color([])
        ax.set_title(f"Aproximaciones – t = {i} min")
        return (scat, vline_100)

    
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames), interval=1000 / fps, blit=True
    )
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        print(f"GIF guardado en {out_path}")
    finally:
        plt.close(fig)

#! ---------------------------------------EJERCICIO 3 ------------------------------------------
#! Estimar la proba de que llegen exactamente 5 aviones en una hora 
#! ---------------------------------------------------------------------------------------------
def proba5_aviones(lam = 1/60, horas = 10000, seed = 42):
    
    t0, t1 = 0, horas*60
    spawns = tiempos_de_spawn(lam, t0, t1, seed)

    contador = [0]*horas #Crea un array de 0's de size horas
    for t in spawns:
        hora = t//60  #Como la funcion tiempos de spawn esta en minutos, quiero pasarlo a horas con division entera (redondea hacia abajo)
        if hora < horas:
            contador[hora] +=1
    
    proba5 = sum(1 for c in contador if c == 5) / horas #Hago el promedio sumando 1 si y solo si el contador en la hora c = 5 (es decir llegaron 5 aviones en esa hora)
    mediaArribos = sum(contador)/horas #solo sirve como una especie de sanity check

    return proba5, mediaArribos


#!Main con warmup para arrancar desde un estado mas interesante
if __name__ == "__main__":
    # λ = prob de aparición por minuto
    lam = 0.5       # ≈ 6 aviones/hora de arribo al horizonte
    warmup = 120     # arrancar 2 horas “antes” (minutos relativos)
    t_obs = 1080     # ventana de observación (06:00–24:00)

    frames, lanes = simular(
        lam=lam,
        t_inicio=-warmup,  
        t_final=t_obs,
        seed=42,
    )

    save_gif_frames(frames, lanes, out_path="sim_ej1.gif", fps=10)

    #? Main ejercicio 3
    lamEj3 = 1/60
    proba5, mediaArribos = proba5_aviones(lamEj3, 50_000, 42)
    print(f"[EJ3] λ={lamEj3:.6f} -> P(#=5 en 1h)≈ {proba5:.6f} | media por hora≈ {mediaArribos:.4f}")



#TODO Esto iria todo en las slides creo

#! ---------------------------------------EJERCICIO 2 ------------------------------------------
#! Si el promedio de arribos es 1 avion por hora → lambda = 18/1080 = 1/60 = 0,016 (6 periodico) 
#! ---------------------------------------------------------------------------------------------


#? ---------------------------------------EJERCICIO 3 ------------------------------------------
#? Verificacion de manera analitica del resultado
#
#   Defino Xi (Bernoulli) como la variable "hubo arribo en el minuto i" con P(X=1) = lambda

#   En una hora hay n = 60 minutos -> el numero de arribos por hora se transforma en N = sum(i=1 hasta 60 de Xi) ~ Binom( p=lambda, n=60 )

#   La proba de que hayan k arribos en una hora es exactamente:
#   P(N = k) = numComb(60, k) * lambda^k * (1-lambda)^60-k 

#   Con lambda = 1/60 y k = 5 entonces;
#   P(N = 5) = numComb(60, 5) * 1/60^5 * (59/60)^55 ~= 0.0027868
#? ---------------------------------------------------------------------------------------------