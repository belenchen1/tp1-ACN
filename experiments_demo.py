# experiments_demo.py
import argparse, os, json, math, statistics
from aep_sim_starter import simulate_day, FREE_FLOW_ETA
import matplotlib.pyplot as plt

def run_many(lam: float, hours: int, seed: int, **kwargs):
    """Corre simulaciones independientes, concatenando días de 18 horas (06:00–24:00)."""
    results = []
    rnd_seed = seed
    for i in range(hours):
        res = simulate_day(lam=lam, seed=rnd_seed, **kwargs)
        results.append(res.metrics)
        rnd_seed += 1
    return results

def ci95(xs):
    if len(xs) < 2: return (statistics.mean(xs), 0.0)
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) < 30 else statistics.stdev(xs)
    # z ~ 1.96
    return (m, 1.96 * s / math.sqrt(len(xs)))

def plot_example(timeline, outdir):
    plt.figure()
    plt.plot(timeline, [1]*len(timeline), 'o')
    plt.xlabel("Minuto del día")
    plt.ylabel("Aterrizajes (eventos)")
    plt.title("Timeline de aterrizajes (ejemplo)")
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, "timeline_ejemplo.png")
    plt.savefig(fn, bbox_inches="tight")
    print(f"Guardado {fn}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.1, help="Probabilidad por minuto de aparición a 100 nm")
    ap.add_argument("--hours", type=int, default=100, help="Cantidad de días simulados (06–24)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--windy", action="store_true", help="Activa 10% de go-arounds (ítem 5)")
    ap.add_argument("--closure", type=str, default=None, help='Hora local de cierre sorpresivo, ej "12:00"')
    args = ap.parse_args()

    closure_minute = None
    if args.closure:
        hh, mm = map(int, args.closure.split(":"))
        # convertir a minutos desde 06:00 (t=0)
        closure_minute = (hh*60 + mm) - 6*60
        closure_minute = max(0, min(18*60, closure_minute))

    outdir = "./out"
    os.makedirs(outdir, exist_ok=True)

    metrics_list = run_many(
        lam=args.lam,
        hours=args.hours,
        seed=args.seed,
        windy=args.windy,
        closure_minute=closure_minute,
    )

    # recoger vectores y mostrar CI95
    spawned = [m["spawned"] for m in metrics_list]
    landed = [m["landed"] for m in metrics_list]
    diverted = [m["diverted"] for m in metrics_list]
    cong_rate = [m["congestion_rate_per_flight"] for m in metrics_list]
    delays = [m["avg_delay_min"] for m in metrics_list]

    for name, arr in [
        ("spawned", spawned),
        ("landed", landed),
        ("diverted", diverted),
        ("congestion_rate_per_flight", cong_rate),
        ("avg_delay_min", delays),
    ]:
        m, e = ci95(arr)
        print(f"{name:28s} = {m:.3f} ± {e:.3f}  (CI95)")

    # guardar un json con agregados
    summary = {
        "lambda_per_min": args.lam,
        "FREE_FLOW_ETA_min": FREE_FLOW_ETA,
        "hours": args.hours,
        "ci95": {
            "spawned": ci95(spawned),
            "landed": ci95(landed),
            "diverted": ci95(diverted),
            "congestion_rate_per_flight": ci95(cong_rate),
            "avg_delay_min": ci95(delays),
        },
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ejemplo de gráfico
    # (solo para el último día corrido; para la entrega, generen gráficos mejores)
    res_example = simulate_day(lam=args.lam, seed=args.seed, windy=args.windy, closure_minute=closure_minute)
    plot_example(res_example.timeline_landings, outdir)

if __name__ == "__main__":
    main()
