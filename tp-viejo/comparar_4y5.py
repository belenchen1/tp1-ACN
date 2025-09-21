# comparar_ej4_ej5.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths a los CSV que generan tus scripts
EJ4_CSV = "ej4_outputs/ej4_resultados.csv"
EJ5_CSV = "ej5_outputs/ej5_resultados.csv"  # ajustá si guardaste en otro lado (p.ej. "ej5_resultados.csv")
OUT_DIR = "ej45_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré {path}. Corré primero el script que lo genera.")
    return pd.read_csv(path)

def align_on_lambda(df4, df5):
    """Deja sólo los λ en común y ordena."""
    common = sorted(set(df4["lambda_per_min"]).intersection(set(df5["lambda_per_min"])))
    df4 = df4[df4["lambda_per_min"].isin(common)].sort_values("lambda_per_min").reset_index(drop=True)
    df5 = df5[df5["lambda_per_min"].isin(common)].sort_values("lambda_per_min").reset_index(drop=True)
    return df4, df5

def save_overlay_plot(df4, df5, ycol, ylabel, out_path, with_ci=True):
    """
    ycol: e.g. 'avg_delay_min_mean' | 'congestion_rate_mean' | 'divert_rate_mean'
    Se asume que existen columnas *_ci_low y *_ci_high con mismo prefijo.
    """
    base = ycol.replace("_mean", "")
    x = df4["lambda_per_min"].values
    y4 = df4[ycol].values
    y5 = df5[ycol].values

    plt.figure()
    if with_ci and f"{base}_ci_low" in df4 and f"{base}_ci_low" in df5:
        y4err = np.vstack([y4 - df4[f"{base}_ci_low"].values,
                           df4[f"{base}_ci_high"].values - y4])
        y5err = np.vstack([y5 - df5[f"{base}_ci_low"].values,
                           df5[f"{base}_ci_high"].values - y5])
        plt.errorbar(x, y4, yerr=y4err, fmt='o-', label="Ej4 (sin interrupciones)")
        plt.errorbar(x, y5, yerr=y5err, fmt='s--', label="Ej5 (con interrupciones 10%)")
    else:
        plt.plot(x, y4, 'o-', label="Ej4 (sin interrupciones)")
        plt.plot(x, y5, 's--', label="Ej5 (con interrupciones 10%)")

    plt.xlabel("λ (arribos por minuto)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs λ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_delta_plot(df4, df5, ycol, ylabel, out_path):
    """
    Dibuja la diferencia (Ej5 - Ej4) con IC95% propagando errores (aprox. independencia).
    """
    base = ycol.replace("_mean", "")
    x = df4["lambda_per_min"].values
    y4 = df4[ycol].values
    y5 = df5[ycol].values
    delta = y5 - y4

    # SE ≈ (ci_high - ci_low) / (2*1.96)
    se4 = (df4[f"{base}_ci_high"].values - df4[f"{base}_ci_low"].values) / (2*1.96)
    se5 = (df5[f"{base}_ci_high"].values - df5[f"{base}_ci_low"].values) / (2*1.96)
    se_delta = np.sqrt(se4**2 + se5**2)
    ci_low = delta - 1.96*se_delta
    ci_high = delta + 1.96*se_delta

    yerr = np.vstack([delta - ci_low, ci_high - delta])

    plt.figure()
    plt.errorbar(x, delta, yerr=yerr, fmt='o-')
    plt.axhline(0, color='k', linewidth=0.8, linestyle=':')
    plt.xlabel("λ (arribos por minuto)")
    plt.ylabel(f"Δ {ylabel}  (Ej5 − Ej4)")
    plt.title(f"Impacto de interrupciones (10%) en {ylabel}")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def print_side_by_side(df4, df5, ycol, label):
    base = ycol.replace("_mean", "")
    print(f"\n--- {label} ---")
    for i, row in df4.iterrows():
        lam = row["lambda_per_min"]
        r4 = row[ycol]
        r5 = df5.loc[i, ycol]
        ci4 = (row[f"{base}_ci_low"], row[f"{base}_ci_high"])
        ci5 = (df5.loc[i, f"{base}_ci_low"], df5.loc[i, f"{base}_ci_high"])
        print(f"λ={lam:>5.2f} | Ej4: {r4:.3f} [{ci4[0]:.3f}, {ci4[1]:.3f}]  |  Ej5: {r5:.3f} [{ci5[0]:.3f}, {ci5[1]:.3f}]  | Δ={r5-r4:+.3f}")

def main():
    df4 = load_df(EJ4_CSV)
    df5 = load_df(EJ5_CSV)
    df4, df5 = align_on_lambda(df4, df5)

    # Overlays
    save_overlay_plot(df4, df5, "avg_delay_min_mean", "Atraso promedio (min)",
                      os.path.join(OUT_DIR, "delay_overlay.png"))
    save_overlay_plot(df4, df5, "congestion_rate_mean", "Frecuencia de congestión",
                      os.path.join(OUT_DIR, "congestion_overlay.png"))
    save_overlay_plot(df4, df5, "divert_rate_mean", "Tasa de desvío (por arribo)",
                      os.path.join(OUT_DIR, "diverts_overlay.png"))

    # Deltas (impacto Ej5−Ej4)
    save_delta_plot(df4, df5, "avg_delay_min_mean", "Atraso promedio (min)",
                    os.path.join(OUT_DIR, "delay_delta.png"))
    save_delta_plot(df4, df5, "congestion_rate_mean", "Frecuencia de congestión",
                    os.path.join(OUT_DIR, "congestion_delta.png"))
    save_delta_plot(df4, df5, "divert_rate_mean", "Tasa de desvío (por arribo)",
                    os.path.join(OUT_DIR, "diverts_delta.png"))

    # Resumen en consola
    print_side_by_side(df4, df5, "avg_delay_min_mean", "Atraso promedio (min)")
    print_side_by_side(df4, df5, "congestion_rate_mean", "Frecuencia de congestión")
    print_side_by_side(df4, df5, "divert_rate_mean", "Tasa de desvío (por arribo)")

    print(f"\nListo. PNGs en: {OUT_DIR}")

if __name__ == "__main__":
    main()
