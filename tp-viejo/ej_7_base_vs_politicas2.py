from main import TraficoAviones as TA_Base, VELOCIDADES as VELOCIDADES_BASE, DAY_START as DS, DAY_END as DE
from ej_7_politica2b import TraficoAviones as TA_2b, VELOCIDADES as VELOCIDADES_2b, DAY_START as DS2b, DAY_END as DE2b # por riesgo de desvio
from ej_7_politica2a import TraficoAviones as TA_2a, VELOCIDADES as VELOCIDADES_2a, DAY_START as DS2a, DAY_END as DE2a # FIFO turnaround

import matplotlib.pyplot as plt
import numpy as np
import math
import os
from main import mins_a_aep

def correr_una_vez(TA_cls, VELOCIDADES, lam: float, seed: int, day_start: int, day_end: int):
   ctrl = TA_cls(seed=seed)
   apariciones = set(ctrl.bernoulli_aparicion(lam, t0=day_start, t1=day_end))
   landed_time = {}
   ideal_abs = {}
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
   delays = [max(0.0, t_land - ideal_abs[aid]) for aid, t_land in landed_time.items()]
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

def correr_varias(TA_cls, VELOCIDADES, lam: float, n_runs: int, base_seed: int, day_start: int, day_end: int):
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

def _yerr_or_none(arr):
   return None if np.isnan(arr).all() else arr

def graficar(lambdas, base_hist, pol2a_hist, pol2b_hist, outdir="resultados"):
   os.makedirs(outdir, exist_ok=True)
   L = np.array(lambdas, dtype=float)
   # Extraigo series
   b_delay  = np.array([h["delay_mean"] for h in base_hist])
   b_delay_se = np.array([h["delay_mean_se"] for h in base_hist])
   p2a_delay  = np.array([h["delay_mean"] for h in pol2a_hist])
   p2a_delay_se = np.array([h["delay_mean_se"] for h in pol2a_hist])
   p2b_delay  = np.array([h["delay_mean"] for h in pol2b_hist])
   p2b_delay_se = np.array([h["delay_mean_se"] for h in pol2b_hist])

   b_div   = np.array([h["div_rate"] for h in base_hist]) * 100.0
   b_div_se= np.array([h["div_rate_se"] for h in base_hist]) * 100.0
   p2a_div   = np.array([h["div_rate"] for h in pol2a_hist]) * 100.0
   p2a_div_se= np.array([h["div_rate_se"] for h in pol2a_hist]) * 100.0
   p2b_div   = np.array([h["div_rate"] for h in pol2b_hist]) * 100.0
   p2b_div_se= np.array([h["div_rate_se"] for h in pol2b_hist]) * 100.0

   b_land  = np.array([h["n_landed"] for h in base_hist])
   p2a_land  = np.array([h["n_landed"] for h in pol2a_hist])
   p2b_land  = np.array([h["n_landed"] for h in pol2b_hist])

   # 1) Atraso medio ± SE
   plt.figure(figsize=(7,5))
   plt.errorbar(L, b_delay, yerr=_yerr_or_none(b_delay_se), marker='o', linestyle='-', label='Base')
   plt.errorbar(L, p2a_delay, yerr=_yerr_or_none(p2a_delay_se), marker='o', linestyle='-', label='Política 2a')
   plt.errorbar(L, p2b_delay, yerr=_yerr_or_none(p2b_delay_se), marker='o', linestyle='-', label='Política 2b')
   plt.xlabel('λ (apariciones por minuto)')
   plt.ylabel('Atraso medio (min)')
   plt.title('Atraso medio vs λ (±EE)')
   plt.grid(True, alpha=0.3)
   plt.legend()
   plt.tight_layout()
   plt.savefig(os.path.join(outdir, "delay_vs_lambda_p2.png"), dpi=160)

   # 2) % Desvíos ± SE
   plt.figure(figsize=(7,5))
   plt.errorbar(L, b_div, yerr=_yerr_or_none(b_div_se), marker='o', linestyle='-', label='Base')
   plt.errorbar(L, p2a_div, yerr=_yerr_or_none(p2a_div_se), marker='o', linestyle='-', label='Política 2a')
   plt.errorbar(L, p2b_div, yerr=_yerr_or_none(p2b_div_se), marker='o', linestyle='-', label='Política 2b')
   plt.xlabel('λ (apariciones por minuto)')
   plt.ylabel('% de desvíos')
   plt.title('Tasa de desvíos vs λ (±EE)')
   plt.grid(True, alpha=0.3)
   plt.legend()
   plt.tight_layout()
   plt.savefig(os.path.join(outdir, "desvios_vs_lambda_p2.png"), dpi=160)

   # 3) Aterrizajes promedio
   plt.figure(figsize=(7,5))
   width = 0.25
   idx = np.arange(len(L))
   plt.bar(idx - width, b_land, width, label='Base')
   plt.bar(idx, p2a_land, width, label='Política 2a')
   plt.bar(idx + width, p2b_land, width, label='Política 2b')
   plt.xticks(idx, [f"{l:.2f}" for l in L])
   plt.xlabel('λ (apariciones por minuto)')
   plt.ylabel('Aterrizajes por día')
   plt.title('Aterrizajes promedio vs λ')
   plt.grid(True, axis='y', alpha=0.3)
   plt.legend()
   plt.tight_layout()
   plt.savefig(os.path.join(outdir, "landings_vs_lambda_p2.png"), dpi=160)

   print(f"\nGráficos guardados en: {os.path.abspath(outdir)}")
   print(" - delay_vs_lambda_p2.png")
   print(" - desvios_vs_lambda_p2.png")
   print(" - landings_vs_lambda_p2.png")

def comparar(lambdas, n_runs=5, base_seed=42):
   assert DS == DS2a and DE == DE2a and DS == DS2b and DE == DE2b, "DAY_START/DAY_END difieren entre módulos."
   t0, t1 = DS, DE
   base_hist = []
   pol2a_hist = []
   pol2b_hist = []
   print("\n==== COMPARACIÓN: BASE vs POLÍTICA 2a vs POLÍTICA 2b ====\n")
   print(f"(n_runs por punto = {n_runs}, día = [{t0}, {t1}))\n")
   header = f"{'λ':>5} | {'delay_base (±SE)':>22} | {'delay_2a (±SE)':>22} | {'delay_2b (±SE)':>22} | {'div%_base (±SE)':>20} | {'div%_2a (±SE)':>20} | {'div%_2b (±SE)':>20} | {'landed_base':>12} | {'landed_2a':>12} | {'landed_2b':>12}"
   print(header)
   print("-"*len(header))
   for lam in lambdas:
      base = correr_varias(TA_Base, VELOCIDADES_BASE, lam, n_runs, base_seed, t0, t1)
      pol2a = correr_varias(TA_2a, VELOCIDADES_2a, lam, n_runs, base_seed+10_000, t0, t1)
      pol2b = correr_varias(TA_2b, VELOCIDADES_2b, lam, n_runs, base_seed+20_000, t0, t1)
      base_hist.append(base)
      pol2a_hist.append(pol2a)
      pol2b_hist.append(pol2b)
      def fmt(val, se, scale=1.0):
         if math.isnan(val): return "nan"
         return f"{val*scale:6.2f} ± {se*scale:4.2f}" if not math.isnan(se) else f"{val*scale:6.2f}"
      line = (
         f"{lam:5.2f} | "
         f"{fmt(base['delay_mean'], base['delay_mean_se']):>22} | "
         f"{fmt(pol2a['delay_mean'], pol2a['delay_mean_se']):>22} | "
         f"{fmt(pol2b['delay_mean'], pol2b['delay_mean_se']):>22} | "
         f"{fmt(base['div_rate'], base['div_rate_se'], 100):>20} | "
         f"{fmt(pol2a['div_rate'], pol2a['div_rate_se'], 100):>20} | "
         f"{fmt(pol2b['div_rate'], pol2b['div_rate_se'], 100):>20} | "
         f"{base['n_landed']:12.1f} | "
         f"{pol2a['n_landed']:12.1f} | "
         f"{pol2b['n_landed']:12.1f}"
      )
      print(line)
   graficar(lambdas, base_hist, pol2a_hist, pol2b_hist)
   print("\nNotas:")
   print("- delay_*: atraso medio (min) respecto al mínimo físico desde 100 nm (bandas), ±EE entre corridas.")
   print("- div%_*: tasa de desvíos sobre vuelos que aparecieron ese día, ±EE entre corridas.")
   print("- landed_*: aterrizajes promedio por día.")

if __name__ == "__main__":
   lambdas = [0.02, 0.1, 0.2, 0.5, 1]
   comparar(lambdas, n_runs=2000, base_seed=123)