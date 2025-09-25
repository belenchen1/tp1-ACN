
# VizViento.py

from __future__ import annotations
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from TraficoAEPViento import TraficoAEPViento
from Constants import DAY_END, DAY_START, VELOCIDADES

# ---------------------- utilitarios de dibujo ----------------------
def preparar_bandas(ax, ymin=-2.0, ymax=2.0):
    """Pinta las franjas de distancia con colores suaves usando VELOCIDADES."""
    colores = ["#e8f7fa", "#e6f8e0", "#fff2cc", "#f3e6ff", "#ffe6e6"]
    for i, (lo, hi, _, _) in enumerate(VELOCIDADES):
        x0 = lo
        x1 = 120.0 if hi == float("inf") else hi
        ax.axvspan(x0, x1, color=colores[i % len(colores)], alpha=0.45, ec=None)
        etiqueta_hi = "100+" if hi == float("inf") else str(int(hi))
        cx = (x0 + (100 if hi == float("inf") else x1)) / 2.0
        ax.text(cx, ymax - 0.25, f"{int(lo)}-{etiqueta_hi}nm",
                ha="center", va="top", fontsize=9, color="#555")
    ax.axvline(0.0, color="black", lw=2)   
    ax.axvline(100.0, color="black", lw=1) 

def mm_to_hhmm(mins: int) -> str:
    h = (mins // 60) % 24
    m = mins % 60
    return f"{h:02d}:{m:02d}"

def generar_apariciones(ctrl, lam_per_min: float, t0: int, t1: int) -> set[int]:
    """Bernoulli por minuto con el RNG del controlador."""
    apar = set()
    for t in range(t0, t1):
        if ctrl.rng.random() < lam_per_min:
            apar.add(t)
    return apar

# ---------------------- simulación con viento ----------------------
def run_live_viento(lambda_per_min: float = 0.10,
                    seed: int = 42,
                    speed: int = 1,
                    p_goaround: float = 0.10,
                    final_threshold_nm: float = 20.0,
                    titulo: str = "simulacion de aproximacion de aviones - aep (viento / go-around)"):
    """
    Visual en vivo para el escenario con viento (usa TraficoAEPViento).
    - lambda_per_min: tasa de llegada (min^-1)
    - seed: semilla reproducible
    - speed: pasos de simulación por frame (1..10)
    - p_goaround: prob. de go-around cuando entra a la zona final
    - final_threshold_nm: umbral de distancia para evaluar go-around
    """
    speed = max(1, int(speed))
    ctrl = TraficoAEPViento(seed=seed, p_goaround=p_goaround, final_threshold_nm=final_threshold_nm)
    apariciones = generar_apariciones(ctrl, lambda_per_min, DAY_START, DAY_END)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(titulo, fontsize=14, pad=12)
    ax.set_xlim(0, 120); ax.invert_xaxis()
    ax.set_ylim(-2, 2)
    ax.set_xlabel("distancia al aeropuerto (millas nauticas)")
    ax.grid(True, linestyle=":", alpha=0.35)

    preparar_bandas(ax, ymin=-2, ymax=2)

    # Capas de puntos
    approach_scatter    = ax.scatter([], [], s=80, color="#2ecc71", zorder=3)  # approach (verde)
    turnaround_scatter  = ax.scatter([], [], s=80, color="#e74c3c", zorder=3)  # turnaround (rojo)
    interrupted_scatter = ax.scatter([], [], s=80, color="#f39c12", zorder=3)  # go-around (naranja)
    diverted_scatter    = ax.scatter([], [], s=60, color="#7f8c8d", zorder=3)  # desviados (gris)

    # Etiquetas por punto
    labels = []

    # Cajas de info
    info_izq = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#3d85c6", alpha=0.9)
    )
    info_der = ax.text(
        0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#ccf2d1", edgecolor="#6aa84f", alpha=0.9)
    )

    state = {"t": DAY_START, "paused": False, "speed": speed}

    def clear_labels():
        for txt in labels:
            txt.remove()
        labels.clear()

    def collect_positions():
        # Activos (approach)
        x_app = [ctrl.planes[aid].distancia_nm for aid in getattr(ctrl, "activos", [])]
        y_app = [1.0] * len(x_app)

        # Turnaround (alejándose)
        x_turn = [ctrl.planes[aid].distancia_nm for aid in getattr(ctrl, "turnaround", [])]
        y_turn = [0.0] * len(x_turn)

        # Go-around (interrupted)
        x_int = [ctrl.planes[aid].distancia_nm for aid in getattr(ctrl, "interrupted", [])] if hasattr(ctrl, "interrupted") else []
        y_int = [0.5] * len(x_int)

        # Inactivos (solo desviados para mostrar en “pileta” inferior)
        diverted_x, diverted_y = [], []
        y_dn = -1.6
        step = 0.12
        for aid in getattr(ctrl, "inactivos", []):
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
        info_izq.set_text(
            f"dia: 1\n"
            f"hora: {mm_to_hhmm(tnow)}\n"
            f"aeropuerto: abierto\n"
            f"lambda: {lambda_per_min:.4f}"
        )
        aterr = sum(1 for a in getattr(ctrl, "inactivos", []) if ctrl.planes[a].estado == "landed")
        divs  = sum(1 for a in getattr(ctrl, "inactivos", []) if ctrl.planes[a].estado == "diverted")
        texto_der = (
            f"aviones activos: {len(getattr(ctrl, 'activos', []))}\n"
            f"total generados: {len(ctrl.planes)}\n"
            f"aterrizados: {aterr}\n"
            f"desviados: {divs}\n"
            f"p(go-around): {p_goaround:.2f}\n"
            f"umbral final: {final_threshold_nm:.0f} nm"
        )
        info_der.set_text(texto_der)

    def add_labels():
        for aid in getattr(ctrl, "activos", []):
            av = ctrl.planes[aid]
            labels.append(ax.annotate(
                f"ID: {aid}\n{int(round(av.velocidad_kts))}kt",
                (av.distancia_nm, 1.0),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85)
            ))
        for aid in getattr(ctrl, "turnaround", []):
            av = ctrl.planes[aid]
            labels.append(ax.annotate(
                f"ID: {aid}\n{int(round(av.velocidad_kts))}kt",
                (av.distancia_nm, 0.0),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85)
            ))
        if hasattr(ctrl, "interrupted"):
            for aid in ctrl.interrupted:
                av = ctrl.planes[aid]
                labels.append(ax.annotate(
                    f"ID: {aid}\n{int(round(av.velocidad_kts))}kt",
                    (av.distancia_nm, 0.5),
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
            approach_scatter, turnaround_scatter, interrupted_scatter, diverted_scatter,
            info_izq, info_der, *labels
        ]

    anim = FuncAnimation(fig, step_frame, interval=100, blit=True)
    plt.tight_layout()
    plt.show()

# ---------------------- ejecución directa ----------------------
if __name__ == "__main__":
    # Valores por defecto; podés ajustar por línea de comandos si querés
    run_live_viento(lambda_per_min=0.10, seed=42, speed=1, p_goaround=0.15, final_threshold_nm=20.0)
