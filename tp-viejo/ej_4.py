from __future__ import annotations
import math, os, random
from dataclasses import dataclass
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import TraficoAviones, DAY_START, DAY_END, velocidad_por_distancia, knots_to_nm_per_min, VELOCIDADES


#Calcula el tiempo de llegada desde su aparicion hasta aterrizar en el caso que no hay congestion
#es decir a vmax por banda
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

BASELINE_TIME_MIN = mins_a_aep(100,300)



#--------------------------------------------------------------------------------------------

#Clase Metricas para luego comparar resultados para distintos niveles de lambda
@dataclass
class Metricas:
    lam: float
    arrivals: int
    landed: int
    diverted: int
    minutos_aviones: int
    minutos_aviones_cong: int
    minutos_delay_promedio: float
    delays: List[float]


#------------------------------------------------------------------------------------------

def simular_jornada(ctrl_seed: int, lam_per_min: float) -> Metricas:
    #Simulacion de una jornada de 18 horas para un λ

    ctrl = TraficoAviones(seed = ctrl_seed)
    arribos_set = set(ctrl.bernoulli_aparicion(lam_per_min, t0=DAY_START, t1=DAY_END))
    arribos = len(arribos_set)

    minutos_aviones = 0
    minutos_aviones_cong = 0
    landed_prev: Set[int] = set()
    delays: List[float] = []

    def landed_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "landed"}
    
    def diverted_set() -> Set[int]:
        return {aid for aid in ctrl.inactivos if ctrl.planes[aid].estado == "diverted"}


    for t in range(DAY_START, DAY_END):
        ctrl.step(t, aparicion=(t in arribos_set))

        #congestion: minutos_aviones con velociad < vmax en approach
        for aid in ctrl.activos:
            av = ctrl.planes[aid]
            vmin, vmax = velocidad_por_distancia(av.distancia_nm)
            minutos_aviones += 1
            if av.velocidad_kts < vmax - 1e-9:
                minutos_aviones_cong += 1

        #nuevos aterrizados: atraso = t_landig - (t_aparicion + T_base)
        now_landed = landed_set()
        new_landed = now_landed - landed_prev
        for aid in new_landed:
            av = ctrl.planes[aid]
            llegada_esperada = av.aparicion_min + BASELINE_TIME_MIN
            delays.append(t - llegada_esperada)
        landed_prev = now_landed
    
    diverted = len(diverted_set())
    landed = len(landed_prev)
    minutos_delay_promedio = float(np.mean(delays)) if delays else float('nan')

    return Metricas(
        lam = lam_per_min,
        arrivals = arribos,
        landed = landed,
        diverted = diverted,
        minutos_aviones = minutos_aviones,
        minutos_aviones_cong = minutos_aviones_cong,
        minutos_delay_promedio = minutos_delay_promedio,
        delays = delays
    )



#-------------------------------------------------------------------------------------------
#IC95% para una media (aproximadamente normal)

def ic95_media(sample: List[float]) -> Tuple[float, float]:
    n = len(sample)
    if n == 0:
        return (float('nan'), float('nan'))
    
    media = float(np.mean(sample))
    desvio_estandar = float(np.std(sample, ddof=1)) if n > 1 else 0.0
    error_estandar = desvio_estandar/math.sqrt(n) if n > 0 else float('nan')
    return (media - 1.96 * error_estandar, media + 1.96 * error_estandar)



#------------------------------------------------------------------------------------------
#Corre ejercicio 4 para varios λ y varios dias y guarda resultados

def montecarlo_dias(lams: List[float], dias: int = 100, seed: int = 25) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for lam in lams:
        daily_cong_rates = []
        daily_avg_delays = []
        daily_divert_rates = []
        daily_arrivals = []
        daily_divert_counts = []

        for _ in range(dias):
            dm = simular_jornada(rng.randrange(10**9), lam)

            cong_rate = (dm.minutos_aviones_cong / dm.minutos_aviones) if dm.minutos_aviones > 0 else float('nan')
            daily_cong_rates.append(cong_rate)

            daily_avg_delays.append(dm.minutos_delay_promedio)

            divert_rate = (dm.diverted / dm.arrivals) if dm.arrivals > 0 else float('nan')
            daily_divert_rates.append(divert_rate)

            daily_arrivals.append(dm.arrivals)
            daily_divert_counts.append(dm.diverted)

        
        #promedios + IC95%
        cong_mean = float(np.nanmean(daily_cong_rates))
        cong_ci = ic95_media([x for x in daily_cong_rates if not math.isnan(x)])

        delay_mean = float(np.nanmean(daily_avg_delays))
        delay_ci = ic95_media([x for x in daily_avg_delays if not math.isnan(x)])

        divert_mean = float(np.nanmean(daily_divert_rates))
        divert_ci = ic95_media([x for x in daily_divert_rates if not math.isnan(x)])

        rows.append({
            "lambda_per_min": lam,
            "days": dias,
            "congestion_rate_mean": cong_mean,
            "congestion_rate_ci_low": cong_ci[0],
            "congestion_rate_ci_high": cong_ci[1],
            "avg_delay_min_mean": delay_mean,
            "avg_delay_min_ci_low": delay_ci[0],
            "avg_delay_min_ci_high": delay_ci[1],
            "divert_rate_mean": divert_mean,
            "divert_rate_ci_low": divert_ci[0],
            "divert_rate_ci_high": divert_ci[1],
            "avg_arrivals_per_day": float(np.nanmean(daily_arrivals)),
            "avg_diverted_per_day": float(np.nanmean(daily_divert_counts)),
        })

    return pd.DataFrame(rows)


#------------------------------------------------------------------------------------------
#Graficos 
def save_plot_y_vs_lambda(df: pd.DataFrame, ycol: str, ylabel: str, out_path: str):
    x = df["lambda_per_min"].values
    y = df[ycol].values
    base = ycol.replace("_mean", "")
    low = df.get(f"{base}_ci_low", pd.Series([np.nan]*len(df)))
    high = df.get(f"{base}_ci_high", pd.Series([np.nan]*len(df)))

    plt.figure()
    if not low.isna().all() and not high.isna().all():
        yerr = np.vstack([y - low, high - y])
        plt.errorbar(x, y, yerr=yerr, fmt='o-')
    else:
        plt.plot(x, y, 'o-')
    plt.xlabel("λ (arribos por minuto)")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs λ")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


#---------------------------------------------------------------------------------------
#main
def main():
    LAMBDAS = [0.02, 0.1, 0.2, 0.5, 1.0]
    DAYS = 100           
    SEED = 2025

    df = montecarlo_dias(LAMBDAS, DAYS, SEED)

    os.makedirs("ej4_outputs", exist_ok=True)
    csv_path = os.path.join("ej4_outputs", "ej4_resultados.csv")
    df.to_csv(csv_path, index=False)

    save_plot_y_vs_lambda(df, "congestion_rate_mean", "Frecuencia de congestión (plane-minutes)", os.path.join("ej4_outputs","congestion_vs_lambda.png"))
    save_plot_y_vs_lambda(df, "avg_delay_min_mean", "Atraso promedio (min)", os.path.join("ej4_outputs","delay_vs_lambda.png"))
    save_plot_y_vs_lambda(df, "divert_rate_mean", "Tasa de desvío (por arribo)", os.path.join("ej4_outputs","diverts_vs_lambda.png"))

    # 👇 Agregado: imprimir % de desvíos y atraso medio
    print("\n--- Resultados resumidos por λ ---")
    for _, row in df.iterrows():
        lam = row["lambda_per_min"]
        divert_pct = row["divert_rate_mean"] * 100   # porcentaje de desviados
        avg_delay = row["avg_delay_min_mean"]        # atraso promedio en minutos
        print(f"λ = {lam:.2f} -> {divert_pct:.2f}% desviados | Atraso medio: {avg_delay:.2f} min")

    print("\nListo. Resultados en:", csv_path)


if __name__ == "__main__":
    main()

