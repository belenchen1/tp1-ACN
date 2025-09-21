# barrido_obj_sep.py
# Barrido de OBJ_SEP_BASE en la Política A (metering anticipado)
# Ajustá el import a tu archivo de política:
import ej_7_politica1 as policyA  # <-- si tu archivo se llama distinto, cambialo

import math
import numpy as np
import matplotlib.pyplot as plt

# ------- tiempo ideal (mínimo físico) usando las bandas -------
def tiempo_ideal_desde(d_nm_inicial: float, VELOCIDADES) -> float:
    """Tiempo mínimo (min) para ir desde d_nm_inicial a 0 usando SIEMPRE vmax de cada banda."""
    eps = 1e-9
    d = float(d_nm_inicial)
    t_min = 0.0

    def banda_para(x):
        for low, high, vmin, vmax in VELOCIDADES:
            if low <= x < high:
                return low, high, vmin, vmax
        return VELOCIDADES[0]

    while d > 0.0:
        x = max(d - eps, 0.0)
        low, high, vmin, vmax = banda_para(x)
        tramo = d - max(low, 0.0)
        if tramo <= 0.0:
            tramo = min(d, 1e-6)
        t_min += tramo / (vmax / 60.0)
        d -= tramo
    return t_min

# ------- una corrida con un valor dado de OBJ_SEP_BASE -------
def correr_una_vez(lam: float, seed: int):
    ctrl = policyA.TraficoAviones(seed=seed)
    apar = set(ctrl.bernoulli_aparicion(lam, t0=policyA.DAY_START, t1=policyA.DAY_END))

    landed_time = {}
    ideal_abs   = {}
    diverted    = set()

    for t in range(policyA.DAY_START, policyA.DAY_END):
        ctrl.step(t, aparicion=(t in apar))
        for aid in list(ctrl.inactivos):
            av = ctrl.planes[aid]
            if av.estado == "landed" and aid not in landed_time:
                landed_time[aid] = t
                ideal_abs[aid] = av.aparicion_min + tiempo_ideal_desde(100.0, policyA.VELOCIDADES)
            elif av.estado == "diverted":
                diverted.add(aid)

    # métricas
    delays = []
    for aid, t_land in landed_time.items():
        delay = max(0.0, t_land - ideal_abs[aid])
        delays.append(delay)

    delay_mean = float(np.mean(delays)) if delays else float("nan")
    delay_std  = float(np.std(delays, ddof=1)) if len(delays) >= 2 else float("nan")
    delay_se   = delay_std / math.sqrt(len(delays)) if len(delays) >= 2 else float("nan")

    n_spawned = len(ctrl.planes)
    n_div     = len(diverted)
    n_land    = len(landed_time)
    div_rate  = n_div / n_spawned if n_spawned else float("nan")

    return {
        "delay_mean": delay_mean,
        "delay_se": delay_se,
        "div_rate": div_rate,
        "n_landed": n_land,
        "n_spawned": n_spawned,
    }

# ------- promedio sobre varias corridas para un OBJ_SEP_BASE -------
def correr_varias(lam: float, obj_sep_base: float, n_runs: int, base_seed: int):
    policyA.OBJ_SEP_BASE = obj_sep_base  # <-- aquí seteamos el parámetro a testear

    mets = []
    for k in range(n_runs):
        mets.append(correr_una_vez(lam, base_seed + k))

    def agg(key):
        vals = [m[key] for m in mets if not (isinstance(m[key], float) and math.isnan(m[key]))]
        if not vals:
            return float("nan"), float("nan")
        mean = float(np.mean(vals))
        se   = float(np.std(vals, ddof=1)/math.sqrt(len(vals))) if len(vals) >= 2 else float("nan")
        return mean, se

    delay_mean, delay_se = agg("delay_mean")
    div_rate, div_se     = agg("div_rate")
    landed, landed_se    = agg("n_landed")
    spawned, _           = agg("n_spawned")

    return {
        "obj": obj_sep_base,
        "delay_mean": delay_mean, "delay_se": delay_se,
        "div_rate": div_rate, "div_se": div_se,
        "landed": landed, "landed_se": landed_se,
        "spawned": spawned,
    }

# ------- barrido principal y plots -------
def barrer_y_graficar(lambdas, objetos, n_runs=6, base_seed=123):
    """
    lambdas: lista de λ a testear (p.ej. [0.08, 0.10, 0.12])
    objetos: lista de OBJ_SEP_BASE (p.ej. [5.0, 5.5, 6.0, 6.5, 7.0])
    """
    resultados = {lam: [] for lam in lambdas}

    print("\n== Barrido de OBJ_SEP_BASE en Política A ==")
    print(f"(n_runs={n_runs}, día=[{policyA.DAY_START}, {policyA.DAY_END}))\n")
    print(f"{'λ':>5} {'OBJ':>5} | {'Delay ±EE (min)':>16} | {'Desvíos% ±EE':>13} | {'Aterrizajes':>12}")
    print("-"*60)
    for lam in lambdas:
        for obj in objetos:
            r = correr_varias(lam, obj, n_runs, base_seed)
            resultados[lam].append(r)
            delay_txt = f"{r['delay_mean']:5.2f} ± {r['delay_se']:4.2f}" if not math.isnan(r['delay_se']) else f"{r['delay_mean']:5.2f}"
            div_txt   = f"{100*r['div_rate']:5.2f} ± {100*r['div_se']:4.2f}" if not math.isnan(r['div_se']) else f"{100*r['div_rate']:5.2f}"
            print(f"{lam:5.2f} {obj:5.2f} | {delay_txt:>16} | {div_txt:>13} | {r['landed']:12.1f}")

    # --- Gráfico 1: Atraso medio vs OBJ_SEP_BASE ---
    plt.figure(figsize=(8,5))
    for lam in lambdas:
        xs = [r["obj"] for r in resultados[lam]]
        ys = [r["delay_mean"] for r in resultados[lam]]
        es = [r["delay_se"] for r in resultados[lam]]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=f"λ={lam}")
    plt.xlabel("OBJ_SEP_BASE (min)")
    plt.ylabel("Atraso medio (min)")
    plt.title("Atraso medio vs OBJ_SEP_BASE (±EE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Gráfico 2: % de desvíos vs OBJ_SEP_BASE ---
    plt.figure(figsize=(8,5))
    for lam in lambdas:
        xs = [r["obj"] for r in resultados[lam]]
        ys = [100*r["div_rate"] for r in resultados[lam]]
        es = [100*r["div_se"] for r in resultados[lam]]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=f"λ={lam}")
    plt.xlabel("OBJ_SEP_BASE (min)")
    plt.ylabel("% de desvíos")
    plt.title("Tasa de desvíos vs OBJ_SEP_BASE (±EE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Gráfico 3: Aterrizajes promedio vs OBJ_SEP_BASE ---
    plt.figure(figsize=(8,5))
    width = 0.12
    offsets = np.linspace(-width*(len(lambdas)-1)/2, width*(len(lambdas)-1)/2, len(lambdas))
    for off, lam in zip(offsets, lambdas):
        xs = np.array([r["obj"] for r in resultados[lam]], float) + off
        ys = [r["landed"] for r in resultados[lam]]
        plt.bar(xs, ys, width=width, label=f"λ={lam}")
    plt.xlabel("OBJ_SEP_BASE (min)")
    plt.ylabel("Aterrizajes por día")
    plt.title("Aterrizajes vs OBJ_SEP_BASE")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    # --- Gráfico 4: Trade-off (delay vs desvíos) para cada λ ---
    plt.figure(figsize=(8,6))
    for lam in lambdas:
        xs = [100*r["div_rate"] for r in resultados[lam]]
        ys = [r["delay_mean"] for r in resultados[lam]]
        labs = [f"{r['obj']:.1f}" for r in resultados[lam]]
        plt.plot(xs, ys, "-o", label=f"λ={lam}")
        for x, y, txt in zip(xs, ys, labs):
            plt.annotate(txt, (x, y), xytext=(5,5), textcoords="offset points", fontsize=9)
    plt.xlabel("% de desvíos")
    plt.ylabel("Atraso medio (min)")
    plt.title("Trade-off: atraso vs desvíos (número = OBJ_SEP_BASE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # elegí qué λ y qué OBJ_SEP_BASE querés barrer
    lambdas = [0.02, 0.1, 0.2, 0.5, 1]
    objetos = [5.0, 5.5, 6.0, 6.5, 7.0]   # probá valores >5
    barrer_y_graficar(lambdas, objetos, n_runs=100, base_seed=2025)
