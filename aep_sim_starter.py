# aep_sim_starter.py
# MIT License
# Motor de simulación en tiempo discreto (dt = 1 minuto) para el TP1 AEP.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math, random

MINUTE = 1.0

def nm_to_km(nm: float) -> float: return nm * 1.852
def knots_to_kmh(k: float) -> float: return k * 1.852
def knots_to_nm_per_min(k: float) -> float: return k / 60.0  # nm/min

# ---- Parámetros del problema ----
RUNWAY_MIN_SEP = 4.0  # min
RUNWAY_TARGET_SEP = 5.0  # min (buffer)
BACKTRACK_SPEED = 200.0  # kts (hacia atrás)
REJOIN_GAP_REQ = 10.0  # min
AEP_OPEN_MIN = 6*60
AEP_CLOSE_MIN = 24*60  # medianoche (medimos desde 00:00)

SPEED_BANDS = [
    # (dist_min_inclusive, dist_max_exclusive, vmin, vmax)  en nmi y kts
    (100.0, float('inf'), 300.0, 500.0),  # >100 nm
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

def band_for_distance(d_nm: float) -> Tuple[float, float]:
    for lo, hi, vmin, vmax in SPEED_BANDS:
        if lo <= d_nm < hi:
            return vmin, vmax
    # si está más allá de 100 nm:
    return SPEED_BANDS[0][2], SPEED_BANDS[0][3]

@dataclass
class Aircraft:
    id: int
    t_spawn: float  # minuto en que aparece a 100 nm
    d_nm: float = 100.0
    speed_kts: float = 300.0
    status: str = "approach"  # approach | backtrack | diverted | landed | goaround
    t_landed: Optional[float] = None
    baseline_eta: Optional[float] = None  # ETA sin congestión (para demora base)
    ever_congested: bool = False
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, d_nm, speed_kts)

    def allowed_band(self) -> Tuple[float, float]:
        return band_for_distance(self.d_nm)

    def step(self, dt_min: float, open_runway: bool) -> None:
        """Avanza posición en nm según `speed_kts` y actualiza estado de aterrizaje si corresponde."""
        if self.status == "approach":
            self.d_nm = max(0.0, self.d_nm - knots_to_nm_per_min(self.speed_kts) * dt_min)
            if self.d_nm == 0.0 and open_runway:
                self.status = "landed"
        elif self.status == "backtrack":
            # se aleja del aeropuerto
            self.d_nm += knots_to_nm_per_min(BACKTRACK_SPEED) * dt_min
            if self.d_nm > 100.0:
                self.status = "diverted"
        # otros estados no mueven (diverted/landed)
        self.history.append((self.t_spawn + len(self.history)*dt_min, self.d_nm, self.speed_kts))

@dataclass
class SimResult:
    flights: List[Aircraft]
    metrics: Dict[str, float]
    timeline_landings: List[float]

def bernoulli_arrivals(lam_per_min: float, t0: int, t1: int, rng: random.Random) -> List[int]:
    """Devuelve minutos (enteros) en [t0, t1) donde aparece 1 avión a 100 nm (proceso Bernoulli por minuto)."""
    times = []
    for t in range(t0, t1):
        if rng.random() < lam_per_min:
            times.append(t)
    return times

def free_flow_eta_minutes() -> float:
    """Tiempo mínimo desde 100 nm a 0 nm volando siempre al vmax permitido de cada banda (aprox)."""
    # integramos por bandas:
    segs = [
        (100.0, 50.0, 300.0),  # 100->50 a 300 kts
        (50.0, 15.0, 250.0),
        (15.0, 5.0, 200.0),
        (5.0, 0.0, 150.0),
    ]
    total_min = 0.0
    for d0, d1, vmax in segs:
        dist = (d0 - d1)  # nm
        v_nm_per_min = vmax / 60.0
        total_min += dist / v_nm_per_min
    return total_min

FREE_FLOW_ETA = free_flow_eta_minutes()

def simulate_day(
    lam: float,
    day_minutes: int = (AEP_CLOSE_MIN - AEP_OPEN_MIN),
    seed: int = 42,
    windy: bool = False,
    closure_minute: Optional[int] = None,
    closure_duration: int = 30,
) -> SimResult:
    """
    Simula desde 06:00 hasta medianoche (t = 0..day_minutes).
    - lam: probabilidad por minuto de nuevo avión.
    - windy: cada aterrizaje tiene 10% de go-around (reinserción sencilla).
    - closure_minute: si no es None, cierra pista [closure_minute, closure_minute+closure_duration).
    """
    rng = random.Random(seed)
    t0, t1 = 0, day_minutes
    spawns = bernoulli_arrivals(lam, t0, t1, rng)

    flights: List[Aircraft] = []
    for i, t in enumerate(spawns):
        ac = Aircraft(id=i+1, t_spawn=t, d_nm=100.0, speed_kts=300.0, status="approach")
        ac.baseline_eta = t + FREE_FLOW_ETA
        flights.append(ac)

    timeline_landings: List[float] = []
    last_landing_time = -1e9

    # simulación minuto a minuto
    for t in range(t0, t1+1):
        # pista abierta?
        open_runway = True
        if closure_minute is not None and closure_minute <= t < closure_minute + closure_duration:
            open_runway = False

        # ordenar por proximidad para aplicar reglas de separación
        approaching = [f for f in flights if f.status in ("approach", "backtrack")]
        approaching.sort(key=lambda f: f.d_nm)

        # velocidad por defecto = vmax por banda (o backtrack fijo)
        leader_speed = None
        for f in approaching:
            if f.status == "backtrack":
                f.speed_kts = BACKTRACK_SPEED
                continue

            vmin, vmax = f.allowed_band()
            desired = vmax  # sin congestión
            # separación con el de adelante (si existe)
            if leader_speed is not None:
                # estimar gap temporal tosco: diferencia de distancias divididas por speeds (heurística)
                # (aprox rápido para decidir si hay riesgo; lo refinado implicaría predecir ETAs por integración)
                lead = leader  # definido en loop anterior
                # tiempo restante estimado (min) = d / (v_nm/min)
                t_self = f.d_nm / (desired/60.0)
                t_lead = lead.d_nm / (lead.speed_kts/60.0)
                gap = (t_self - t_lead)

                if gap < RUNWAY_MIN_SEP:
                    # regla: el follower baja 20 kts vs el líder hasta lograr >=5 min
                    candidate = min(desired, max(vmin, lead.speed_kts - 20.0))
                    if candidate < vmin:
                        # congestión fuerte: entra en backtrack
                        f.status = "backtrack"
                        f.speed_kts = BACKTRACK_SPEED
                        f.ever_congested = True
                    else:
                        f.speed_kts = candidate
                        f.ever_congested = True
                        # reeval gap tosco; si sigue corto, lo dejamos para el próximo minuto
                else:
                    f.speed_kts = desired
            else:
                f.speed_kts = desired

            leader_speed = f.speed_kts
            leader = f  # para el próximo

        # avanzar todos
        for f in flights:
            prev_d = f.d_nm
            f.step(dt_min=1.0, open_runway=open_runway)

        # registrar aterrizajes con regla de separación real en la pista
        for f in flights:
            if f.status == "landed" and f.t_landed is None:
                # aplica la separación en la pista (si está muy cerca del anterior, el aterrizaje se difiere)
                landing_time = float(t)
                if timeline_landings and landing_time - timeline_landings[-1] < RUNWAY_MIN_SEP:
                    # no puede aterrizar aún; forzamos un pequeño "hold" de 1 min
                    # (en un modelo más fino esto sería un go-around; aquí lo tratamos como espera corta)
                    f.status = "approach"
                    f.d_nm = max(0.5, f.d_nm)  # lo devolvemos levemente arriba
                else:
                    # OK, aterriza
                    f.t_landed = landing_time
                    timeline_landings.append(landing_time)
                    last_landing_time = landing_time
                    # ¿go-around estocástico (viento)?
                    if windy and random.random() < 0.1:
                        # vuelve a 6 nm y lo marcamos como 'goaround' temporalmente
                        f.status = "goaround"
                        f.d_nm = 6.0
                        f.speed_kts = 180.0
                        f.t_landed = None  # todavía no había aterrizado
                    else:
                        f.status = "landed"

        # procesar goarounds: se vuelven "approach" al minuto siguiente
        for f in flights:
            if f.status == "goaround":
                f.status = "approach"

        # si está backtracking y pasó 100 nm => desvío
        # (la lógica avanzada de reingreso en gap ≥10 min queda como TODO)
        # TODO: Implementar reingreso cuando exista gap ≥10 min en `timeline_landings` futuro.

    # métricas
    landed = [f for f in flights if f.t_landed is not None]
    diverted = [f for f in flights if f.status == "diverted"]
    congested = [f for f in flights if f.ever_congested]

    delays = []
    for f in landed:
        if f.baseline_eta is not None:
            delays.append(max(0.0, f.t_landed - f.baseline_eta))

    metrics = {
        "spawned": len(flights),
        "landed": len(landed),
        "diverted": len(diverted),
        "congestion_rate_per_flight": (len(congested) / max(1, len(flights))),
        "avg_delay_min": (sum(delays)/len(delays)) if delays else 0.0,
    }
    return SimResult(flights=flights, metrics=metrics, timeline_landings=timeline_landings)
