# viz_live_viento.py
from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from TraficoAEPViento import TraficoAEPViento
from Constants import DAY_END, DAY_START, VELOCIDADES

def preparar_bandas(ax, ymin=-2.0, ymax=2.0):
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

def run_live_viento(lambda_per_min: float = 0.10, seed: int = 42, speed: int = 1):
    ctrl = TraficoAEPViento(seed=seed)
    apariciones = set(ctrl.bernoulli_aparicion(lambda_per_min, t0=DAY_START, t1=DAY_END))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("simulación de aproximación con viento (go-around)", fontsize=14, pad=12)
    ax.set_xlim(0, 120); ax.invert_xaxis()
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("distancia al aeropuerto (millas náuticas)")
    ax.grid(True, linestyle=":", alpha=0.35)

    preparar_bandas(ax, ymin=-2.5, ymax=2.5)

    approach_scatter    = ax.scatter([], [], s=80, color="#2ecc71", zorder=3)
    turnaround_scatter  = ax.scatter([], [], s=80, color="#e74c3c", zorder=3)
    interrupted_scatter = ax.scatter([], [], s=80, color="#f39c12", zorder=3)
    diverted_scatter    = ax.scatter([], [], s=60, color="#7f8c8d", zorder=3)

    labels = []

    info_izq = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#3d85c6", alpha=0.9)
    )
    info_der = ax.text(
        0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#6aa84f", alpha=0.9)
    )

    state = {"t": DAY_START, "paused": False, "speed": max(1, int(speed))}

    def clear_labels():
        for txt in labels:
            txt.remove()
        labels.clear()

    def collect_positions():
        x_app  = [ctrl.planes[aid].distancia_nm for aid in ctrl.activos]
        y_app  = [1.5] * len(x_app)

        x_turn = [ctrl.planes[aid].distancia_nm for aid in ctrl.turnaround]
        y_turn = [0.5] * len(x_turn)

        x_int  = [ctrl.planes[aid].distancia_nm for aid in ctrl.interrupted]
        y_int  = [-0.5] * len(x_int)

        diverted_x, diverted_y = [], []
        y_dn = -2.0
        step = 0.12
        for aid in ctrl.inactivos:
            av = ctrl.planes[aid]
            if av.estado == "diverted":
                diverted_x.append(110.0)
                diverted_y.append(y_dn)
                y_dn += step

        app_xy  = np.column_stack([x_app, y_app])   if x_app  else np.empty((0, 2))
        turn_xy = np.column_stack([x_turn, y_turn]) if x_turn else np.empty((0, 2))
        int_xy  = np.column_stack([x_int, y_int])   if x_int  else np.empty((0, 2))
        div_xy  = np.column_stack([diverted_x, diverted_y]) if diverted_x else np.empty((0, 2))

        return app_xy, turn_xy, int_xy, div_xy

    def update_info_boxes(tnow: int):
        texto_izq = (
            f"día: 1\n"
            f"hora: {mm_to_hhmm(tnow)}\n"
            f"aeropuerto: abierto\n"
            f"lambda: {lambda_per_min:.4f}"
        )
        info_izq.set_text(texto_izq)

        aterr = sum(1 for a in ctrl.inactivos if ctrl.planes[a].estado == "landed")
        divs  = sum(1 for a in ctrl.inactivos if ctrl.planes[a].estado == "diverted")
        # contamos todos los que alguna vez estuvieron en interrupted
        total_interrupted = sum(1 for av in ctrl.planes.values() if getattr(av, "ever_interrupted", False))
        texto_der = (
            f"activos: {len(ctrl.activos)}\n"
            f"turnaround: {len(ctrl.turnaround)}\n"
            f"interrupted actuales: {len(ctrl.interrupted)}\n"
            f"interrupted totales: {total_interrupted}\n"
            f"total generados: {len(ctrl.planes)}\n"
            f"aterrizados: {aterr}\n"
            f"desviados: {divs}"
        )
        info_der.set_text(texto_der)

    def add_labels():
        for aid in ctrl.activos + ctrl.turnaround + ctrl.interrupted:
            av = ctrl.planes[aid]
            y = 1.5 if aid in ctrl.activos else (0.5 if aid in ctrl.turnaround else -0.5)
            labels.append(ax.annotate(
                f"ID: {aid}\n{int(round(av.velocidad_kts))}kt",
                (av.distancia_nm, y),
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
            fig.savefig("frame_viento.png", dpi=150)
            print("PNG guardado:", "frame_viento.png")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def step_frame(_):
        if state["paused"]:
            return []

        for _ in range(state["speed"]):
            if state["t"] >= DAY_END:
                break
            ctrl.step(state["t"], aparicion=(state["t"] in apariciones))

            # marcar si se interrumpió
            for aid in ctrl.interrupted:
                ctrl.planes[aid].ever_interrupted = True

            # imprimir aterrizajes nuevos
            for aid in ctrl.inactivos:
                av = ctrl.planes[aid]
                if av.estado == "landed" and not hasattr(av, "_reported"):
                    print(f"Avión {aid} aterrizó a las {mm_to_hhmm(av.aterrizaje_min)}")
                    av._reported = True  # flag para no volver a imprimirlo

            state["t"] += 1

        app_xy, turn_xy, int_xy, div_xy = collect_positions()
        approach_scatter.set_offsets(app_xy)
        turnaround_scatter.set_offsets(turn_xy)
        interrupted_scatter.set_offsets(int_xy)
        diverted_scatter.set_offsets(div_xy)

        clear_labels()
        add_labels()
        update_info_boxes(state["t"])

        return [
            approach_scatter,
            turnaround_scatter,
            interrupted_scatter,
            diverted_scatter,
            info_izq,
            info_der,
            *labels,
        ]


    # más rápido que antes (interval = 120 ms)
    anim = FuncAnimation(
        fig,
        step_frame,
        interval=600,            # ya lento
        blit=False,              # <- clave para listas de artistas variables
        cache_frame_data=False,  # <- evita cache no acotado
        save_count=(DAY_END - DAY_START + 2)  # opcional, por si guardás
)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_live_viento(lambda_per_min=0.50, seed=42, speed=1)
