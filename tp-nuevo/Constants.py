
#Constants.py

MINUTE = 1.0
SEPARACION_MINIMA = 5.0  # separación deseada a ambos lados (min)
SEPARACION_PELIGRO = 4.0 # debajo de esto, hay conflicto (actuar fuerte)
VEL_TURNAROUND = 200.0   # kts alejándose
MAX_DIVERTED_DISTANCE = 100.0      # si se aleja más de 100 nm -> diverted
DAY_START = 0
DAY_END = 1080 


# Parámetros de la Política A (metering anticipado)
OBJ_SEP_BASE = 5.5         # target "cómodo" (>5). Podés barrerlo para ver tradeoff atraso↔desvíos
BUFFER_ANTICIPACION = 0.5  # anticipo para empezar a ajustar un poco antes


# Bandas (dist_lo, dist_hi, vmin, vmax)
VELOCIDADES = [
    (100.0, float('inf'), 300.0, 500.0),
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

