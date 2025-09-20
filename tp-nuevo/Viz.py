# viz_live.py
from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# En Windows, si no abre ventana, descomentá:
# matplotlib.use("TkAgg")

from TraficoAEP import TraficoAviones
from Constants import DAY_END, DAY_START, VELOCIDADES

# ---------------------- helpers de dibujo ----------------------
def preparar_bandas(ax, ymin=-2.0, ymax=2.0):
    # Colores suaves, un poco más intensos (alpha=0.45)
    colores = ["#e8f7fa", "#e6f8e0", "#fff2cc", "#f3e6ff", "#ffe6e6"]
    for i, (lo, hi, _, _) in enumerate(VELOCIDADES):
        x0 = lo
        x1 = 120.0 if hi == float("inf") else hi
        ax.axvspan(x0, x1, color=colores[i % len(colores)], alpha=0.45, ec=None)
        cx = (x0 + (100 if hi == float("inf") else x1)) / 2.0
        ax.text(
            cx, ymax - 0.25, f"{int(lo)}-{('100+' if hi == float('inf') else int(hi))}nm",
            ha="center", va="top", fontsize=9, color="#555"
        )
    ax.axvline(0.0, color="black", lw=2)
    ax.axvline(100.0, color="black", lw=1)

def mm_to_hhmm(mins: int) -> str:
    h = (mins // 60) % 24
    m = mins % 60
    return f"{h:02d}:{m:02d}"

# ---------------------- animación en vivo ----------------------
def run_live(lambda_per_min: float = 0.10, seed: int = 42, speed: int = 1):
    """
    speed = 1 => 1 minuto de simulación por frame
    speed = 5 => 5 minutos por frame (más rápido)
    """
    ctrl = TraficoAviones(seed=seed)
    apariciones = set(ctrl.bernoulli_aparicion(lambda_per_min, t0=DAY_START, t1=DAY_END))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("simulacion de aproximacion de aviones - aep", fontsize=14, pad=12)
    ax.set_xlim(0, 120); ax.invert_xaxis()
    ax.set_ylim(-2, 2)
    ax.set_xlabel("distancia al aeropuerto (millas nauticas)")
    ax.grid(True, linestyle=":", alpha=0.35)

    preparar_bandas(ax, ymin=-2, ymax=2)

    # colecciones de puntos
    approach_scatter   = ax.scatter([], [], s=80, color="#2ecc71", zorder=3)   # approach (verde)
    turnaround_scatter = ax.scatter([], [], s=80, color="#e74c3c", zorder=3)   # turnaround (rojo)
    diverted_scatter   = ax.scatter([], [], s=60, color="#7f8c8d", zorder=3)   # desviados (gris)
    # OJO: no dibujamos "landed" para que desaparezcan

    labels = []

    # cajitas de info
    info_izq = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#3d85c6", alpha=0.9)
    )
    info_der = ax.text(
        0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#ccf2d1", edgecolor="#6aa84f", alpha=0.9)
    )

    # estado de animación
    state = {"t": DAY_START, "paused": False, "speed": max(1, int(speed))}

    def clear_labels():
        for txt in labels:
            txt.remove()
        labels.clear()

    def collect_positions():
        # approach
        x_app = [ctrl.planes[aid].distancia_nm for aid in ctrl.activos]
        y_app = [1.0] * len(x_app)

        # turnaround
        x_turn = [ctrl.planes[aid].distancia_nm for aid in ctrl.turnaround]
        y_turn = [0.0] * len(x_turn)

        # inactivos: SOLO mostramos desviados; los aterrizados se ocultan
        diverted_x, diverted_y = [], []
        y_dn = -1.6
        step = 0.12
        for aid in ctrl.inactivos:
            av = ctrl.planes[aid]
            if av.estado == "diverted":
                diverted_x.append(110.0)
                diverted_y.append(y_dn)
                y_dn += step

        app_xy  = np.column_stack([x_app, y_app])   if x_app  else np.empty((0, 2))
        turn_xy = np.column_stack([x_turn, y_turn]) if x_turn else np.empty((0, 2))
        div_xy  = np.column_stack([diverted_x, diverted_y]) if diverted_x else np.empty((0, 2))

        # landed_xy NO se devuelve (vacío) para que no se dibuje
        return app_xy, turn_xy, div_xy

    def update_info_boxes(tnow: int):
        texto_izq = (
            f"dia: 1\n"
            f"hora: {mm_to_hhmm(tnow)}\n"
            f"aeropuerto: abierto\n"
            f"lambda: {lambda_per_min:.4f}"
        )
        info_izq.set_text(texto_izq)

        aterr = sum(1 for a in ctrl.inactivos if ctrl.planes[a].estado == "landed")
        divs  = sum(1 for a in ctrl.inactivos if ctrl.planes[a].estado == "diverted")
        texto_der = (
            f"aviones activos: {len(ctrl.activos)}\n"
            f"total generados: {len(ctrl.planes)}\n"
            f"aterrizados: {aterr}\n"
            f"desviados: {divs}"
        )
        info_der.set_text(texto_der)

    def add_labels():
        # etiquetas (ID y vel) solo para approach y turnaround (landed ocultos)
        for aid in ctrl.activos:
            av = ctrl.planes[aid]
            labels.append(ax.annotate(
                f"ID: {aid}\n{int(round(av.velocidad_kts))}kt",
                (av.distancia_nm, 1.0),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85)
            ))
        for aid in ctrl.turnaround:
            av = ctrl.planes[aid]
            labels.append(ax.annotate(
                f"ID: {aid}\n{int(round(av.velocidad_kts))}kt",
                (av.distancia_nm, 0.0),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85)
            ))

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "up":
            state["speed"] = min(10, state["speed"] + 1)
        elif event.key == "down":
            state["speed"] = max(1, state["speed"] - 1)
        elif event.key in ("s", "S"):
            fig.savefig("frame_live.png", dpi=150)
            print("PNG guardado:", "frame_live.png")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def step_frame(_):
        if state["paused"]:
            return []

        for _ in range(state["speed"]):
            if state["t"] >= DAY_END:
                break
            ctrl.step(state["t"], aparicion=(state["t"] in apariciones))
            state["t"] += 1

        app_xy, turn_xy, div_xy = collect_positions()
        approach_scatter.set_offsets(app_xy)
        turnaround_scatter.set_offsets(turn_xy)
        diverted_scatter.set_offsets(div_xy)

        clear_labels()
        add_labels()
        update_info_boxes(state["t"])

        return [approach_scatter, turnaround_scatter, diverted_scatter, info_izq, info_der, *labels]

    anim = FuncAnimation(fig, step_frame, interval=60, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_live(lambda_per_min=0.10, seed=42, speed=1)
