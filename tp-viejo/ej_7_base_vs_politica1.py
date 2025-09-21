# comparador.py
# Corre y compara: simulación base vs simulación con Política A, y genera gráficos.

# ====== AJUSTAR ESTO SEGÚN TUS ARCHIVOS ======
from main import TraficoAviones as TA_Base, VELOCIDADES as VELOCIDADES_BASE, DAY_START as DS, DAY_END as DE
from main import *
from ej_7_politica1 import TraficoAviones as TA_Policy, VELOCIDADES as VELOCIDADES_POL, DAY_START as DS2, DAY_END as DE2
# =============================================

import math
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utilidades de tiempo ideal ----------
def mins_a_aep(dist_nm: float, speed_kts: float) -> float:
    ''' cuantos minutos te faltan para llegar a aep si vas a speed_kts constante '''
    t = 0.0
    d = dist_nm
    v = speed_kts
    while d > 0:
        for dist_low, dist_high, vmin, vmax in VELOCIDADES_BASE:
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

# ---------- Motor de una corrida ----------
def correr_una_vez(TA_cls, VELOCIDADES, lam: float, seed: int, day_start: int, day_end: int):
    """
    Ejecuta una simulación con clase TA_cls,
    devuelve métricas agregadas para atraso y desvíos.
    """
    ctrl = TA_cls(seed=seed)
    apariciones = set(ctrl.bernoulli_aparicion(lam, t0=day_start, t1=day_end))

    landed_time = {}   # aid -> minuto aterrizaje
    ideal_abs = {}     # aid -> t ideal absoluto
    diverted_set = set()

    for t in range(day_start, day_end):
        ctrl.step(t, aparicion=(t in apariciones))

        for aid in list(ctrl.inactivos):
            av = ctrl.planes[aid]
            if av.estado == 'landed' and aid not in landed_time:
                landed_time[aid] = t
                ideal_abs[aid] = av.aparicion_min + mins_a_aep(100.0, 300.0)
            elif av.estado == 'diverted':
                diverted_set.add(aid)

    n_land = len(landed_time)
    n_div  = len(diverted_set)

    delays = []
    for aid, t_land in landed_time.items():
        t_ideal = ideal_abs[aid]
        delays.append(max(0.0, t_land - t_ideal))

    delay_mean = float(np.mean(delays)) if delays else float('nan')
    delay_std  = float(np.std(delays, ddof=1)) if len(delays) >= 2 else float('nan')
    delay_se   = delay_std / math.sqrt(len(delays)) if len(delays) >= 2 else float('nan')

    total_spawned = len(ctrl.planes)
    div_rate = n_div / total_spawned if total_spawned > 0 else float('nan')

    return {
        "n_spawned": total_spawned,
        "n_landed": n_land,
        "n_diverted": n_div,
        "div_rate": div_rate,
        "delay_mean": delay_mean,
        "delay_se": delay_se,
    }

# ---------- Multiple corridas para estimar error ----------
def correr_varias(TA_cls, VELOCIDADES, lam: float, n_runs: int, base_seed: int, day_start: int, day_end: int):
    """
    Corre n_runs con seeds distintos y agrega métricas (promedio + SE entre corridas).
    """
    metrics = []
    for k in range(n_runs):
        seed = base_seed + k
        m = correr_una_vez(TA_cls, VELOCIDADES, lam, seed, day_start, day_end)
        metrics.append(m)

    def agg(key):
        vals = [m[key] for m in metrics if not (isinstance(m[key], float) and math.isnan(m[key]))]
        if not vals:
            return (float('nan'), float('nan'))
        mean = float(np.mean(vals))
        se   = float(np.std(vals, ddof=1)/math.sqrt(len(vals))) if len(vals) >= 2 else float('nan')
        return (mean, se)

    delay_mean, delay_mean_se = agg("delay_mean")
    div_rate_mean, div_rate_se = agg("div_rate")
    n_landed_mean, n_landed_se = agg("n_landed")
    n_div_mean, n_div_se       = agg("n_diverted")
    n_spawned_mean, _          = agg("n_spawned")

    return {
        "delay_mean": delay_mean, "delay_mean_se": delay_mean_se,
        "div_rate": div_rate_mean, "div_rate_se": div_rate_se,
        "n_landed": n_landed_mean, "n_landed_se": n_landed_se,
        "n_diverted": n_div_mean, "n_diverted_se": n_div_se,
        "n_spawned": n_spawned_mean,
    }

# ---------- Gráficos ----------
def _yerr_or_none(arr):
    return None if np.isnan(arr).all() else arr

def graficar(lambdas, base_hist, polA_hist, outdir="resultados"):
    os.makedirs(outdir, exist_ok=True)
    L = np.array(lambdas, dtype=float)

    # Extraigo series
    b_delay  = np.array([h["delay_mean"] for h in base_hist])
    b_delay_se = np.array([h["delay_mean_se"] for h in base_hist])
    p_delay  = np.array([h["delay_mean"] for h in polA_hist])
    p_delay_se = np.array([h["delay_mean_se"] for h in polA_hist])

    b_div   = np.array([h["div_rate"] for h in base_hist]) * 100.0
    b_div_se= np.array([h["div_rate_se"] for h in base_hist]) * 100.0
    p_div   = np.array([h["div_rate"] for h in polA_hist]) * 100.0
    p_div_se= np.array([h["div_rate_se"] for h in polA_hist]) * 100.0

    b_land  = np.array([h["n_landed"] for h in base_hist])
    p_land  = np.array([h["n_landed"] for h in polA_hist])

    # 1) Atraso medio ± SE
    plt.figure(figsize=(7,5))
    plt.errorbar(L, b_delay, yerr=_yerr_or_none(b_delay_se), marker='o', linestyle='-', label='Base')
    plt.errorbar(L, p_delay, yerr=_yerr_or_none(p_delay_se), marker='o', linestyle='-', label='Política A')
    plt.xlabel('λ (apariciones por minuto)')
    plt.ylabel('Atraso medio (min)')
    plt.title('Atraso medio vs λ (±EE)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delay_vs_lambda.png"), dpi=160)

    # 2) % Desvíos ± SE
    plt.figure(figsize=(7,5))
    plt.errorbar(L, b_div, yerr=_yerr_or_none(b_div_se), marker='o', linestyle='-', label='Base')
    plt.errorbar(L, p_div, yerr=_yerr_or_none(p_div_se), marker='o', linestyle='-', label='Política A')
    plt.xlabel('λ (apariciones por minuto)')
    plt.ylabel('% de desvíos')
    plt.title('Tasa de desvíos vs λ (±EE)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "desvios_vs_lambda.png"), dpi=160)

    # 3) Aterrizajes promedio
    plt.figure(figsize=(7,5))
    width = 0.35
    idx = np.arange(len(L))
    plt.bar(idx - width/2, b_land, width, label='Base')
    plt.bar(idx + width/2, p_land, width, label='Política A')
    plt.xticks(idx, [f"{l:.2f}" for l in L])
    plt.xlabel('λ (apariciones por minuto)')
    plt.ylabel('Aterrizajes por día')
    plt.title('Aterrizajes promedio vs λ')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "landings_vs_lambda.png"), dpi=160)

    print(f"\nGráficos guardados en: {os.path.abspath(outdir)}")
    print(" - delay_vs_lambda.png")
    print(" - desvios_vs_lambda.png")
    print(" - landings_vs_lambda.png")

# ---------- Comparador principal ----------
def comparar(lambdas, n_runs=5, base_seed=42):
    assert DS == DS2 and DE == DE2, "DAY_START/DAY_END difieren entre módulos."
    t0, t1 = DS, DE

    base_hist = []
    polA_hist = []

    print("\n==== COMPARACIÓN: BASE vs POLÍTICA A ====\n")
    print(f"(n_runs por punto = {n_runs}, día = [{t0}, {t1}))\n")
    header = f"{'λ':>5} | {'delay_base (±SE)':>22} | {'delay_polA (±SE)':>22} | {'div%_base (±SE)':>20} | {'div%_polA (±SE)':>20} | {'landed_base':>12} | {'landed_polA':>12}"
    print(header)
    print("-"*len(header))

    for lam in lambdas:
        base = correr_varias(TA_Base, VELOCIDADES_BASE, lam, n_runs, base_seed, t0, t1)
        polA = correr_varias(TA_Policy, VELOCIDADES_POL, lam, n_runs, base_seed+10_000, t0, t1)
        base_hist.append(base)
        polA_hist.append(polA)

        def fmt(val, se, scale=1.0):
            if math.isnan(val): return "nan"
            return f"{val*scale:6.2f} ± {se*scale:4.2f}" if not math.isnan(se) else f"{val*scale:6.2f}"

        line = (
            f"{lam:5.2f} | "
            f"{fmt(base['delay_mean'], base['delay_mean_se']):>22} | "
            f"{fmt(polA['delay_mean'], polA['delay_mean_se']):>22} | "
            f"{fmt(base['div_rate'], base['div_rate_se'], 100):>20} | "
            f"{fmt(polA['div_rate'], polA['div_rate_se'], 100):>20} | "
            f"{base['n_landed']:12.1f} | "
            f"{polA['n_landed']:12.1f}"
        )
        print(line)

    graficar(lambdas, base_hist, polA_hist)

    print("\nNotas:")
    print("- delay_*: atraso medio (min) respecto al mínimo físico desde 100 nm (bandas), ±EE entre corridas.")
    print("- div%_*: tasa de desvíos sobre vuelos que aparecieron ese día, ±EE entre corridas.")
    print("- landed_*: aterrizajes promedio por día.")

if __name__ == "__main__":
    # Elegí los λ y cuántas corridas por punto
    lambdas = [0.02, 0.1, 0.2, 0.5, 1]
    comparar(lambdas, n_runs=1000, base_seed=123)
