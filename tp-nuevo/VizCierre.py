# viz_live_cierre.py
from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from TraficoAEPCierre import TraficoAEPCerrado, AEPCerrado
from Constants import VELOCIDADES, DAY_START, DAY_END

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

def run_live_cierre(lambda_per_min: float = 0.10, seed: int = 42, speed_fast: int = 6):
    # Cierre en el mismo día (ejemplo: 03:00–03:30)
    closure = AEPCerrado(start_min=DAY_START + 20, dur_min=30)
    ctrl = TraficoAEPCerrado(closure=closure, seed=seed)

    # Apariciones para TODO el día, desde minuto 0
    apariciones = set(ctrl.bernoulli_aparicion(lambda_per_min, t0=DAY_START, t1=DAY_END))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Simulación de aproximación con cierre temporal de AEP (día completo)", fontsize=14, pad=12)
    ax.set_xlim(0, 120); ax.invert_xaxis()
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("Distancia al aeropuerto (millas náuticas)")
    ax.grid(True, linestyle=":", alpha=0.35)

    preparar_bandas(ax, ymin=-2.0, ymax=2.0)

    # scatter plots
    approach_scatter   = ax.scatter([], [], s=80, color="#2ecc71", zorder=3)
    turnaround_scatter = ax.scatter([], [], s=80, color="#e74c3c", zorder=3)
    diverted_scatter   = ax.scatter([], [], s=60, color="#7f8c8d", zorder=3)

    labels = []

    # cajitas de información
    info_izq = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#3d85c6", alpha=0.9)
    )
    info_der = ax.text(
        0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#6aa84f", alpha=0.9)
    )

    state = {"t": DAY_START, "paused": False}

    def clear_labels():
        for txt in labels:
            txt.remove()
        labels.clear()

    def collect_positions():
        x_app  = [ctrl.planes[aid].distancia_nm for aid in ctrl.activos]
        y_app  = [1.0] * len(x_app)

        x_turn = [ctrl.planes[aid].distancia_nm for aid in ctrl.turnaround]
        y_turn = [0.0] * len(x_turn)

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

        return app_xy, turn_xy, div_xy

    def update_info_boxes(tnow: int):
        texto_izq = (
            f"Hora: {mm_to_hhmm(tnow)}\n"
            f"Aeropuerto: {'CERRADO' if closure.is_closed(tnow) else 'Abierto'}\n"
            f"λ: {lambda_per_min:.4f}"
        )
        info_izq.set_text(texto_izq)

        aterr = sum(1 for a in ctrl.inactivos if ctrl.planes[a].estado == "landed")
        divs  = sum(1 for a in ctrl.inactivos if ctrl.planes[a].estado == "diverted")
        texto_der = (
            f"Activos: {len(ctrl.activos)}\n"
            f"Turnaround: {len(ctrl.turnaround)}\n"
            f"Total generados: {len(ctrl.planes)}\n"
            f"Aterrizados: {aterr}\n"
            f"Desviados: {divs}"
        )
        info_der.set_text(texto_der)

    def add_labels():
        for aid in ctrl.activos + ctrl.turnaround:
            av = ctrl.planes[aid]
            y = 1.0 if aid in ctrl.activos else 0.0
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
        elif event.key in ("s", "S"):
            fig.savefig("frame_cierre.png", dpi=150)
            print("PNG guardado:", "frame_cierre.png")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def step_frame(_):
        if state["paused"]:
            return []

        # Avanza siempre; durante cierre 1 min/frame, sino speed_fast min/frame
        steps = 1 if closure.is_closed(state["t"]) else max(1, int(speed_fast))

        for _ in range(steps):
            if state["t"] >= DAY_END:
                break
            ctrl.step(state["t"], aparicion=(state["t"] in apariciones))
            state["t"] += 1

        # imprimir aterrizajes nuevos
        for aid in ctrl.inactivos:
            av = ctrl.planes[aid]
            if av.estado == "landed" and not hasattr(av, "_printed"):
                print(f"Avión {aid} aterrizó a las {mm_to_hhmm(int(av.aterrizaje_min))}")
                av._printed = True

        # actualizar gráficos
        app_xy, turn_xy, div_xy = collect_positions()
        approach_scatter.set_offsets(app_xy)
        turnaround_scatter.set_offsets(turn_xy)
        diverted_scatter.set_offsets(div_xy)

        clear_labels()
        add_labels()
        update_info_boxes(state["t"])

        return [approach_scatter, turnaround_scatter, diverted_scatter, info_izq, info_der, *labels]

    # Intervalo del frame (ms). Bajalo si quieres ver más fluido.
    anim = FuncAnimation(fig, step_frame, interval=600, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # speed_fast controla cuántos minutos por frame cuando está ABIERTO
    run_live_cierre(lambda_per_min=0.50, seed=42, speed_fast=1)
