# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Simulación de aproximaciones a AEP — CÓDIGO ORIGINAL **IGUAL** PERO con un
# ÚNICO cambio: la lógica de REINGRESO desde 'turnaround' ahora busca un HUECO
# GLOBAL ≥ 10 min en la cola (entre cualquier par consecutivo de aviones), en
# lugar de mirar sólo al antiguo líder (id+1). Todo lo demás se mantiene igual.
# 
# Pedido del alumno/a: "COMENTALO MUCHO MÁS, línea por línea" → agrego comentarios
# extensivos explicando QUÉ hace cada línea y POR QUÉ.
# -----------------------------------------------------------------------------

from __future__ import annotations  # Permite anotar tipos de clases aún no definidas (forward refs)

# ---------------------------
# IMPORTS (librerías externas)
# ---------------------------
import matplotlib                   # Núcleo de Matplotlib para gráficos
import numpy as np                  # Numpy para arrays/operaciones vectoriales (lo usamos en el scatter)
from dataclasses import dataclass, field  # dataclass para definir "Avion"; 'field' no se usa aquí
from typing import List, Dict, Optional, Tuple  # Tipado estático opcional (sólo informativo)
import math, random                # 'random' para arribos Bernoulli; 'math' no se usa pero se mantiene igual
import matplotlib.pyplot as plt     # API de Matplotlib estilo MATLAB para graficar
matplotlib.use("Agg")               # Backend no interactivo (genera archivos: PNG/GIF). Mantengo orden del original
from matplotlib.lines import Line2D  # Para construir ítems de leyenda manualmente
import matplotlib.animation as animation  # Para crear animaciones (GIF) con FuncAnimation


# --------------------------------------------------
# CONVERSIÓN DE UNIDADES (helpers súper sencillos)
# --------------------------------------------------
# Nota: 1 nudo (kt) = 1 milla náutica por hora (nm/h). 1 nm = 1.852 km.

def nm_to_km(nm: float) -> float: return nm * 1.852            # Pasa millas náuticas a km

def knots_to_kmh(k: float) -> float: return k * 1.852          # Pasa nudos a km/h

def knots_to_nm_per_min(k: float) -> float: return k / 60.0    # Pasa nudos a nm por minuto (kts = nm/h)


# --------------------------------------------------
# PARÁMETROS GLOBALES DEL PROBLEMA (constantes)
# --------------------------------------------------
MINUTE = 1.0                                # Duración de un "tick" de simulación (1 minuto)
separacion_minima = 4.0                     # Umbral de peligro: gap < 4 min ⇒ hay que frenar/actuar
separacion_target = 5.0                     # Buffer deseado: gap ≥ 5 min ⇒ estoy "cómodo"
velocidad_reversa = 200.0                   # Velocidad al irse "hacia atrás" (turnaround)
apertura_aep_min = 6*60                     # Inicio de la ventana observada: 06:00 en minutos desde 00:00
cierre_aep_min = 24*60                      # Cierre de la ventana: 24:00 en minutos
minutos_del_dia = cierre_aep_min - apertura_aep_min  # Duración de la ventana (1080 min = 18 horas)
x_max = 120.0                               # Límite superior del eje X en la visual (0..120 nm)

# Tabla de BANDAS DE VELOCIDAD permitida según distancia a AEP
# Cada tupla: (dist_min_inclusive, dist_max_exclusive, vmin, vmax)
# Distancias en millas náuticas (nm), velocidades en nudos (kts)
velocidades = [
    (100.0, float('inf'), 300.0, 500.0),  # Más allá de 100 nm (fuera del tubo controlado): 300–500 kts permitidos
    (50.0, 100.0, 250.0, 300.0),          # Entre 50 y 100 nm
    (15.0, 50.0, 200.0, 250.0),           # Entre 15 y 50 nm
    (5.0, 15.0, 150.0, 200.0),            # Entre 5 y 15 nm
    (0.0, 5.0, 120.0, 150.0),             # Entre 0 y 5 nm (tramo final)
]


# --------------------------------------------------
# FUNCIONES ÚTILES (cálculos de política y tiempos)
# --------------------------------------------------

def velocidad_por_distancia(d_nm: float) -> Tuple[float, float]:
    """Devuelve (vmin, vmax) según en qué banda cae la distancia d_nm.
    Recorremos la tabla 'velocidades' y elegimos la primera cuyo rango contenga a d_nm.
    """
    for lo, hi, vmin, vmax in velocidades:  # Itero por cada banda (rango de distancia)
        if lo <= d_nm < hi:                 # Si d_nm cae en [lo, hi)
            return vmin, vmax               # Devuelvo límites de velocidad permitida en esa banda
    # Si por alguna razón d_nm > 100 nm, caigo en la banda de 300–500 kts
    return velocidades[0][2], velocidades[0][3]


def eta_min(dist_nm: float, speed_kts: float) -> float:
    """ETA (min) a pista si mantuvieras speed_kts constante desde dist_nm.
    speed_kts/60 = nm/min; dist_nm / (nm/min) = min. Si speed<=0 ⇒ infinito.
    """
    return float('inf') if speed_kts <= 0 else dist_nm / (speed_kts / 60.0)


def gap_minutos(self_dist: float, self_speed: float,
                lead_dist: float, lead_speed: float) -> float:
    """Gap temporal (min) = ETA(seguidor) - ETA(líder), usando velocidades dadas.
    Si el resultado es positivo ⇒ el seguidor llegaría DESPUÉS (hay separación).
    """
    return eta_min(self_dist, self_speed) - eta_min(lead_dist, lead_speed)


# --------------------------------------------------
# CLASE Avion: estado + regla de decisión por minuto (step)
# --------------------------------------------------
@dataclass
class Avion:
    id: int                                 # Identificador entero. Convenio: líder "nominal" es id+1
    momento_aparicion: float                # Minuto en que aparece a 100 nm (spawn)
    distancia_a_aep: float = 100.0          # Estado: distancia restante a la pista (nm). 0 ⇒ en pista
    velocidad: float = 300.0                # Estado: velocidad actual (kts)
    status: str = "approach"                # Estado discreto: approach|delayed|turnaround|diverted|landed

    def velocidad_permitida(self) -> Tuple[float, float]:
        # Wrapper útil por legibilidad (no altera lógica)
        return velocidad_por_distancia(self.distancia_a_aep)

    def step(self, *,
             cohort: Dict[int, "Avion"],     # Mapa id→Avion para ubicar "líder/vecinos" sin recorrer toda la lista
             dt_min: float = 1.0,            # Tamaño de paso (minutos). Aquí siempre 1
             gap_target_min: float = separacion_target,   # Buffer deseado (5 min)
             gap_minimo_min: float = separacion_minima,   # Umbral peligro (4 min)
             gap_reingreso_min: float = 10.0              # Requisito para reinsertarse tras turnaround
             ) -> None:
        """Aplica la política de control minuto a minuto.
        Casos principales:
          - Normal (no turnaround): si gap<4 ⇒ intento -20 kts vs líder ≥ vmin, si no ⇒ turnaround.
            Si gap≥5 ⇒ approach@vmax. Si 4≤gap<5 ⇒ delayed@vmax.
          - Turnaround: **cambio** pedido: busco HUECO GLOBAL ≥10 min (por ETA) y si lo encuentro
            reingreso a approach@vmax; si no, sigo alejándome a 200 kts; si paso de 100 nm ⇒ diverted.
        """

        # 1) Si ya está fuera de juego (landed/diverted), no hace nada este tick
        if self.status in ("landed", "diverted"):
            return  # corto acá porque no quiero actualizar ni velocidad ni distancia

        # 2) Intento obtener al "líder nominal" (id+1). Si ese líder ya no está (landed/diverted), lo ignoro
        leader = cohort.get(self.id + 1, None)  # O(1) vía diccionario
        if leader and leader.status in ("landed", "diverted"):
            leader = None  # leader inválido, lo trato como si no hubiera

        # 3) Leo límites de velocidad en la banda actual (según mi distancia)
        vmin, vmax = velocidad_por_distancia(self.distancia_a_aep)

        # 4) Caso: estoy en turnaround (voy "hacia atrás" a 200 kts hasta ver hueco)
        if self.status == "turnaround":
            # --- CAMBIO SOLICITADO ---
            # Ahora el reingreso NO mira sólo al viejo líder. Busca un HUECO GLOBAL ≥10 min.

            # 4.a) Construyo lista de aviones "activos" (descarto landed/diverted y a mí mismo).
            activos = [a for a in cohort.values()
                       if a.id != self.id and a.status not in ("landed", "diverted")]

            # 4.b) Si no hay nadie activo, puedo reinsertarme sin conflicto → approach@vmax
            if not activos:
                self.status = "approach"     # vuelvo a la cola
                self.velocidad = vmax         # retomo la velocidad máxima permitida
                # (no hago return: dejo que más abajo se aplique el avance común "hacia adelante")
            else:
                # 4.c) Calculo ETAs de TODOS los activos con sus velocidades actuales (escenario "tal como está")
                cand = [(a, eta_min(a.distancia_a_aep, a.velocidad)) for a in activos]
                cand.sort(key=lambda x: x[1])  # Ordeno por ETA ascendente (orden temporal de llegada)

                # 4.d) Mi ETA hipotética si reingreso a vmax (lo que haría al volver a approach)
                my_eta = eta_min(self.distancia_a_aep, vmax)

                # 4.e) Busco un hueco entre cada par consecutivo (A,B) con:
                #      - Separación ETA ≥ gap_reingreso_min (10 min)
                #      - Y que my_eta caiga DENTRO del hueco (eta(A) ≤ my_eta ≤ eta(B))
                reingresa = False
                for (a1, eta1), (a2, eta2) in zip(cand, cand[1:]):  # recorro pares adyacentes en la cola temporal
                    if (eta2 - eta1) >= gap_reingreso_min and (eta1 <= my_eta <= eta2):
                        # Hueco suficiente y mi ETA cabe dentro → me reinsertaría "entre" A y B
                        self.status = "approach"  # salgo de turnaround
                        self.velocidad = vmax      # aplico la misma velocidad con la que evalué el hueco
                        reingresa = True
                        break

                # 4.f) Si NO encontré hueco global suficiente, sigo en turnaround alejándome a 200 kts
                if not reingresa:
                    self.velocidad = velocidad_reversa                 # 200 kts "hacia atrás"
                    self.distancia_a_aep += knots_to_nm_per_min(self.velocidad) * dt_min  # aumenta distancia
                    if self.distancia_a_aep >= 100.0:                  # si ya me fui del tubo de 100 nm
                        self.status = "diverted"                       # se considera desvío (sale del sistema)
                    return  # IMPORTANTE: ya apliqué desplazamiento; evito el avance común de cierre

        else:
            # 5) Caso: NO estoy en turnaround (operación "normal")
            velDes = vmax  # Política "greedy": intento ir al máximo permitido por mi banda

            if leader is None:
                # 5.a) Sin líder válido ⇒ no hay conflicto inmediato → approach @ vmax
                self.velocidad = velDes
                self.status = "approach"
            else:
                # 5.b) Con líder: calculo gap suponiendo que YO voy a velDes y el líder mantiene su velocidad actual
                gap = gap_minutos(self.distancia_a_aep, velDes, leader.distancia_a_aep, leader.velocidad)

                if gap < gap_minimo_min:
                    # 5.b.i) Estoy "peligrosamente cerca" (<4 min) ⇒ intento frenar 20 kts por debajo del líder
                    nuevaVel = min(velDes, leader.velocidad - 20.0)
                    if nuevaVel < vmin:
                        # 5.b.ii) Para lograr separación tendría que ir < vmin ⇒ NO permitido → turnaround

                        # CHEQUEAR SI ESTA BIEN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        # NO muevas ni desvíes en el mismo tick del cambio de estado
                        self.status = "turnaround"
                        self.velocidad = velocidad_reversa
                        return  # ← dejá que el movimiento/chequeo ocurra en el PRÓXIMO tick, dentro del bloque turnaround

                    else:
                        # 5.b.iii) Pude frenar sin violar vmin ⇒ me quedo en la cola pero marcado como delayed
                        self.velocidad = nuevaVel
                        self.status = "delayed"

                elif gap >= gap_minimo_min:
                    # 5.b.iv) Tengo buffer cómodo (≥4 min) ⇒ approach @ vmax
                    self.status = "approach"
                    self.velocidad = velDes

        # 6) AVANCE COMÚN (aplica si no hice "return" antes: o sea, no en el tramo de turnaround sin reingreso)
        avance_mn = knots_to_nm_per_min(self.velocidad) * dt_min       # nm/min * min = nm
        self.distancia_a_aep = max(0.0, self.distancia_a_aep - avance_mn)  # me acerco (no bajo de 0)
        # Nota: mantenemos el original SIN marcar status="landed" explícito al tocar 0; se capea distancia a 0.

        if self.distancia_a_aep <= 0.0:
            self.distancia_a_aep = 0.0
            self.velocidad = 0.0
            self.status = "landed"
            return  # listo: queda fuera de juego en siguientes ticks

        # si no aterrizó:
        self.distancia_a_aep = max(0.0, self.distancia_a_aep)


# --------------------------------------------------
# LLEGADAS Bernoulli por minuto (spawns)
# --------------------------------------------------
# Generamos de manera estocástica en qué minutos aparece 1 (a lo sumo) avión nuevo.

def tiempos_de_spawn(lam_per_min, t0, t1, seed=42):
    rng = random.Random(seed)  # RNG local con semilla fija (reproducible)
    tiempos = []               # acumulador de minutos donde hubo "éxito"
    for i in range(t0, t1):    # recorro minutos enteros [t0, t1)
        u = rng.random()       # U ~ Uniforme(0,1)
        if u < lam_per_min:    # Éxito con prob λ ⇒ spawnea 1 avión en el minuto i
            tiempos.append(i)
    return tiempos             # lista de minutos con aparición (máx 1 por minuto)


# --------------------------------------------------
# COLORES por estado para la visualización (scatter)
# --------------------------------------------------
# Nota: mantenemos los colores del original (incluye 'tab:pink' para diverted).

def color_estados(estado):
    return {
        "approach":   "tab:blue",
        "delayed":    "tab:orange",
        "turnaround": "tab:red",
        "diverted":   "gray",        # ← usa un color real
        "landed":     "tab:green",   # ← si agregaste el estado landed, dale color
    }.get(estado, "gray")            # ← default SIEMPRE un color válido



# --------------------------------------------------
# SIMULACIÓN PRINCIPAL (time-stepped)
# --------------------------------------------------
# Recorre minuto a minuto: spawnea con Bernoulli, hace step() por avión, guarda frames.

def simular(lam, t_inicio, t_final, seed = 42):
    rng = random.Random(seed)  # Se crea RNG pero no se usa aquí (se mantiene igual al original)

    # Convención de ids: decrecientes (…,-3,-2,-1). El de id más "grande" (-1) suele estar más cerca.
    prox_avion_id = 0  # arrancamos en 0 y vamos decrementando al spawnear

    vuelos: List[Avion] = []          # lista con todos los objetos Avion vivos (y los que ya aterrizaron/desviaron)
    cohort: Dict[int, Avion] = {}     # diccionario id→Avion para acceso O(1)

    spawns = set(tiempos_de_spawn(lam, t_inicio, t_final, seed))  # set de minutos con spawn para testear en O(1)

    # Estructuras sólo para la visual (asignar "carriles" verticales en el gráfico)
    lanes: Dict[int, int] = {}
    lane_counter = 0
    lane_step = 1.2  # separación vertical entre carriles (arbitraria, puramente estética)

    frames = []  # acumulador de fotogramas (xs, ys, cs) por minuto observado

    for t in range(t_inicio, t_final):  # bucle de tiempo discreto (minutos enteros)

        # 1) Spawns: si este minuto pertenece al set, aparece EXACTAMENTE 1 avión a 100 nm
        if t in spawns:
            prox_avion_id -= 1                                   # asigno nuevo id decreciente
            av = Avion(prox_avion_id, t, 100.0, 300.0, "approach")  # objeto Avion inicializado
            vuelos.append(av)                                     # lo agrego a la lista
            cohort[av.id] = av                                    # y al diccionario para lookup
            lanes[av.id] = lane_counter                           # le asigno un "carril" (línea visual) único
            lane_counter += 1                                     # preparo el carril para el próximo

        # 2) Dinámica: actualizo TODOS los aviones, uno por uno, ordenados por id ascendente
        #    (seguidores primero, líderes después), de modo que cada uno decida con el estado
        #    del tick anterior de su líder (actualización "sincrónica" limpia).
        for f in sorted(vuelos, key=lambda a: a.id):
            f.step(cohort=cohort, dt_min=1.0)

        # 3) Captura visual (frames): sólo desde t>=0 (si hubo warmup con t<0 no lo guardo)
        if t >= 0:
            xs, ys, cs = [], [], []                    # listas para distancias, carriles y colores
            for v in vuelos:                           # recorro todos los aviones vivos
                xs.append(max(0.0, min(x_max, v.distancia_a_aep)))  # capeo distancia a [0, x_max] por estética
                ys.append(lanes[v.id] * lane_step)                  # ubico en su carril vertical
                cs.append(color_estados(v.status))                  # color según su estado
            frames.append((xs, ys, cs))               # guardo un frame con los 3 vectores

    return frames, lanes  # devuelvo frames para la animación y el mapeo id→carril


# --------------------------------------------------
# VISUALIZACIÓN: guardar GIF de la evolución de la cola
# --------------------------------------------------
# Construye un scatter animado (x=distancia, y=carril, color=estado).

def save_gif_frames(frames, lanes, out_path="sim_ej1.gif", fps=10):
    fig, ax = plt.subplots(figsize=(10, 5))       # creo figura y ejes con tamaño 10x5 pulgadas
    ax.set_xlim(0, x_max)                         # eje X entre 0 y x_max nm
    ax.set_ylim(-1, max(2, len(lanes)) * 1.2)     # eje Y dinámico según cantidad de carriles
    ax.invert_xaxis()                             # invierto X para que 100→0 vaya de izquierda a derecha
    ax.set_xlabel("Distancia a AEP (nm)")        # etiqueta eje X
    ax.set_ylabel("Pista visual por avión")      # etiqueta eje Y

    scat = ax.scatter([], [])                     # creo un scatter vacío (se llenará en cada frame)
    scat.set_offsets(np.empty((0, 2)))            # inicializo offsets (N×2) a matriz vacía

    # Leyenda con colores fijos por estado (coinciden con los usados en color_estados, salvo diverted aquí es gris)
    legend_handles = [
    Line2D([0],[0], marker='o', linestyle='None', color='tab:blue',   label='approach'),
    Line2D([0],[0], marker='o', linestyle='None', color='tab:orange', label='delayed'),
    Line2D([0],[0], marker='o', linestyle='None', color='tab:red',    label='turnaround'),
    Line2D([0],[0], marker='o', linestyle='None', color='gray',       label='diverted'),
    Line2D([0],[0], marker='o', linestyle='None', color='tab:green',  label='landed'),  # ← nuevo
    ]


    ax.legend(handles=legend_handles, loc="upper right")  # muestro la leyenda arriba a la derecha

    def init():
        # Función de inicialización para FuncAnimation: limpia offsets y colores
        scat.set_offsets(np.empty((0, 2)))  # sin puntos
        scat.set_color([])                  # sin colores
        return (scat,)

    def update(i):
        # Función que dibuja el frame i-ésimo: coloca los puntos y setea colores
        xs, ys, cs = frames[i]             # recupero los vectores del frame i
        if xs:                             # si hay puntos en este frame
            offs = np.column_stack([xs, ys])  # construyo matriz N×2 con columnas (x,y)
            scat.set_offsets(offs)            # actualizo posiciones
            scat.set_color(cs)                # actualizo colores por estado
        else:                             
            scat.set_offsets(np.empty((0, 2)))  # frame sin puntos
            scat.set_color([])
        ax.set_title(f"Aproximaciones – t = {i} min")  # título dinámico con el minuto
        return (scat,)                                 # devuelvo el artista actualizado

    # Creo la animación: llama a update(i) para i=0..len(frames)-1
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames), interval=1000 / fps, blit=True
    )
    try:
        writer = animation.PillowWriter(fps=fps)  # escritor GIF basado en Pillow
        anim.save(out_path, writer=writer)        # guardo en disco el GIF final
        print(f"GIF guardado en {out_path}")
    finally:
        plt.close(fig)  # cierro la figura para liberar memoria/recursos


# --------------------------------------------------
# MAIN: corre simulación con warmup y guarda el GIF
# --------------------------------------------------
if __name__ == "__main__":
    # Parámetros "de demo" (mantenidos IGUAL que en el original)
    lam = 0.05        # Prob de aparición por minuto (≈ 6 aviones/hora)
    warmup = 120     # Arranco 2 horas "antes" para que la cola ya esté funcionando en t=0
    t_obs = 1080     # Ventana visible (06:00–24:00) en minutos

    # Ejecuto simulación completa (incluye warmup negativo hasta t=-warmup)
    frames, lanes = simular(
        lam=lam,
        t_inicio=-warmup,   # min inicial (negativo = warmup)
        t_final=t_obs,      # último minuto excluido de simulación; visibles serán 0..1080
        seed=42,            # semilla para reproducibilidad
    )

    # Guardo animación como GIF (10 fps)
    save_gif_frames(frames, lanes, out_path="sim_ej1.gif", fps=10)
