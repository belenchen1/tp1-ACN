# ===========================
# Métricas y Montecarlo (Ej5)
# ===========================
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import os, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main_ej5 import *

# --- tiempo base (sin congestión) desde 100nm volando a vmax por banda ---
def baseline_time_from_100nm() -> float:
    t = 0.0
    # Integra exactamente por tramos entre 100 -> 0 usando vmax de cada banda
    for dist_low, dist_high, vmin, vmax in VELOCIDADES:
        hi = min(dist_high, 100.0)
        lo = max(dist_low, 0.0)
        if hi <= lo:
            continue
        tramo_nm = hi - lo
        t += tramo_nm / knots_to_nm_per_min(vmax)
    return t

BASELINE_TIME_MIN = baseline_time_from_100nm()


# -----------------------------
# contenedor de métricas diarias
# -----------------------------
@dataclass
class MetricasDia:
    lam: float
    arrivals: int
    landed: int
    diverted: int
    minutos_aviones: int
    minutos_aviones_cong: int
    avg_delay_min: float   # promedio del día (solo landed)
    delays: List[float]    # todos los retrasos del día


# -----------------------------
# simula 1 jornada para un λ dado
# -----------------------------
def simular_jornada(ctrl_seed: int, lam_per_min: float) -> MetricasDia:
    ctrl = TraficoAviones(seed=ctrl_seed)
    arribos_set = set(ctrl.bernoulli_aparicion(lam_per_min, t0=DAY_START, t1=DAY_END))
    arrivals = len(arribos_set)

    minutos_aviones = 0
    minutos_aviones_cong = 0
    landed_prev: Set[int] = set()
    delays: List[float] = []

    def landed_set() -> Set[int]:
        # aterrizados ya movidos a inactivos con estado 'landed'
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "landed"}

    def diverted_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "diverted"}

    for t in range(DAY_START, DAY_END):
        ctrl.step(t, aparicion=(t in arribos_set))

        # minutos de “avión” en approach y congestión (v < vmax de su banda)
        for aid in ctrl.activos:
            av = ctrl.planes[aid]
            vmin, vmax = velocidad_por_distancia(av.distancia_nm)
            minutos_aviones += 1
            if av.velocidad_kts < vmax - 1e-9:
                minutos_aviones_cong += 1

        # nuevos landed -> computo delay contra baseline físico
        now_landed = landed_set()
        nuevos = now_landed - landed_prev
        for aid in nuevos:
            av = ctrl.planes[aid]
            esperada = av.aparicion_min + BASELINE_TIME_MIN
            # usamos el tiempo real guardado en av.aterrizaje_min por tu simulador
            delay = av.aterrizaje_min - esperada if av.aterrizaje_min is not None else (t - esperada)
            delays.append(delay)
        landed_prev = now_landed

    diverted = len(diverted_set())
    landed = len(landed_prev)
    avg_delay_min = float(np.mean(delays)) if delays else float('nan')

    return MetricasDia(
        lam=lam_per_min,
        arrivals=arrivals,
        landed=landed,
        diverted=diverted,
        minutos_aviones=minutos_aviones,
        minutos_aviones_cong=minutos_aviones_cong,
        avg_delay_min=avg_delay_min,
        delays=delays
    )


# -----------------------------
# IC95% de una media (normal aprox)
# -----------------------------
def ic95_media(sample: List[float]) -> Tuple[float, float]:
    sample = [x for x in sample if not math.isnan(x)]
    n = len(sample)
    if n == 0:
        return (float('nan'), float('nan'))
    media = float(np.mean(sample))
    sd = float(np.std(sample, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return (media - 1.96 * se, media + 1.96 * se)


# -----------------------------
# Montecarlo varios λ por N días
# -----------------------------
def montecarlo_dias(lams: List[float], dias: int = 1000, seed: int = 2025) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for lam in lams:
        daily_cong_rates = []   # minutos_aviones_cong / minutos_aviones
        daily_avg_delays = []   # promedio de delays de ese día
        daily_divert_rates = [] # diverted / arrivals
        daily_arrivals = []
        daily_divert_counts = []

        for _ in range(dias):
            md = simular_jornada(rng.randrange(10**9), lam)

            cong_rate = (md.minutos_aviones_cong / md.minutos_aviones) if md.minutos_aviones > 0 else float('nan')
            daily_cong_rates.append(cong_rate)

            daily_avg_delays.append(md.avg_delay_min)

            divert_rate = (md.diverted / md.arrivals) if md.arrivals > 0 else float('nan')
            daily_divert_rates.append(divert_rate)

            daily_arrivals.append(md.arrivals)
            daily_divert_counts.append(md.diverted)

        # promedios + IC95% sobre la distribución diaria
        cong_mean = float(np.nanmean(daily_cong_rates))
        cong_ci = ic95_media(daily_cong_rates)

        delay_mean = float(np.nanmean(daily_avg_delays))
        delay_ci = ic95_media(daily_avg_delays)

        divert_mean = float(np.nanmean(daily_divert_rates))
        divert_ci = ic95_media(daily_divert_rates)

        rows.append({
            "lambda_per_min": lam,
            "days": dias,
            # congestión
            "congestion_rate_mean": cong_mean,
            "congestion_rate_ci_low": cong_ci[0],
            "congestion_rate_ci_high": cong_ci[1],
            # atraso
            "avg_delay_min_mean": delay_mean,
            "avg_delay_min_ci_low": delay_ci[0],
            "avg_delay_min_ci_high": delay_ci[1],
            # desvíos
            "divert_rate_mean": divert_mean,
            "divert_rate_ci_low": divert_ci[0],
            "divert_rate_ci_high": divert_ci[1],
            # extras
            "avg_arrivals_per_day": float(np.nanmean(daily_arrivals)),
            "avg_diverted_per_day": float(np.nanmean(daily_divert_counts)),
        })

    return pd.DataFrame(rows)


# -----------------------------
# Gráficos (con barras de error)
# -----------------------------
def save_plot_y_vs_lambda(df: pd.DataFrame, ycol: str, ylabel: str, out_path: str):
    x = df["lambda_per_min"].values
    y = df[ycol].values
    base = ycol.replace("_mean", "")
    low = df.get(f"{base}_ci_low", pd.Series([np.nan]*len(df)))
    high = df.get(f"{base}_ci_high", pd.Series([np.nan]*len(df)))

    plt.figure()
    if not np.all(np.isnan(low)) and not np.all(np.isnan(high)):
        yerr = np.vstack([y - low.to_numpy(), high.to_numpy() - y])
        plt.errorbar(x, y, yerr=yerr, fmt='o-')
    else:
        plt.plot(x, y, 'o-')
    plt.xlabel("λ (arribos por minuto)")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs λ")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Runner
# -----------------------------
def run_montecarlo_ej5():
    # Elegí la malla de λ que quieras estudiar
    LAMBDAS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    DAYS = 1000
    SEED = 2025

    df = montecarlo_dias(LAMBDAS, DAYS, SEED)

    os.makedirs("ej5_outputs", exist_ok=True)
    csv_path = os.path.join("ej5_outputs", "ej5_resultados.csv")
    df.to_csv(csv_path, index=False)

    save_plot_y_vs_lambda(df, "congestion_rate_mean", "Frecuencia de congestión", os.path.join("ej5_outputs","congestion_vs_lambda.png"))
    save_plot_y_vs_lambda(df, "avg_delay_min_mean", "Atraso promedio (min)", os.path.join("ej5_outputs","delay_vs_lambda.png"))
    save_plot_y_vs_lambda(df, "divert_rate_mean", "Tasa de desvíos", os.path.join("ej5_outputs","diverts_vs_lambda.png"))

    print("Listo. CSV y gráficos en:", os.path.abspath("ej5_outputs"))

if __name__ == "__main__":
    run_montecarlo_ej5()
