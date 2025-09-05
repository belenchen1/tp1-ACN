from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math, random
import matplotlib.pyplot as plt

MINUTE = 1.0

def nm_to_km(nm: float) -> float: return nm * 1.852
def knots_to_kmh(k: float) -> float: return k * 1.852
def knots_to_nm_per_min(k: float) -> float: return k / 60.0  # nm/min

# Parámetros del problema
separacion_minima = 4.0  # min
separacion_target = 5.0  # min (buffer)
velocidad_reversa = 200.0  # kts (hacia atrás)
apertura_aep_min = 6*60
cierre_aep_min = 24*60  # medianoche (medimos desde 00:00)
minutos_del_dia = cierre_aep_min - apertura_aep_min  # 1080 min


velocidades = [
    # (dist_min_inclusive, dist_max_exclusive, vmin, vmax)  en nmi y kts
    (100.0, float('inf'), 300.0, 500.0),  # >100 nm
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

def velocidad_por_distancia(d_nm: float) -> Tuple[float, float]:
    for lo, hi, vmin, vmax in velocidades:
        if lo <= d_nm < hi:
            return vmin, vmax
    # si está más allá de 100 nm:
    return velocidades[0][2], velocidades[0][3]

@dataclass
class Avion:
    id: int
    momento_aparicion: float  # minuto en que aparece a 100 nm
    distancia_a_aep: float = 100.0
    velocidad: float = 300.0
    status: str = "approach"  # approach | backtrack | diverted | landed | goaround
    instante_aterrizaje: Optional[float] = None
    eta_base: Optional[float] = None  # ETA sin congestión (para demora base)
    sufrio_congestion: bool = False
    historial: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, distancia_a_aep, velocidad)

    def velocidad_permitida(self) -> Tuple[float, float]:
        return velocidad_por_distancia(self.distancia_a_aep)

    def step(self, dt_min: float, open_runway: bool) -> None:
        """Avanza posición en nm según `velocidad` y actualiza estado de aterrizaje si corresponde."""
        if self.status == "approach":
            self.distancia_a_aep = max(0.0, self.distancia_a_aep - knots_to_nm_per_min(self.velocidad) * dt_min)
            if self.distancia_a_aep == 0.0 and open_runway:
                self.status = "landed"
        elif self.status == "backtrack":
            # se aleja del aeropuerto
            self.distancia_a_aep += knots_to_nm_per_min(velocidad_reversa) * dt_min
            if self.distancia_a_aep > 100.0:
                self.status = "diverted"
        # otros estados no mueven (diverted/landed)
        self.historial.append((self.momento_aparicion + len(self.historial)*dt_min, self.distancia_a_aep, self.velocidad))


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


@dataclass
class SimResult:
    flights: List[Avion]
    metrics: Dict[str, float]
    timeline_landings: List[float]

def bernoulli_arrivals(lam_per_min: float, t0: int, t1: int, rng: random.Random) -> List[int]:
    """Devuelve minutos (enteros) en [t0, t1) donde aparece 1 avión a 100 nm (proceso Bernoulli por minuto)."""
    times = []
    for t in range(t0, t1):
        if rng.random() < lam_per_min:
            times.append(t)
    return times


def simulate_day(
    lam: float,
    day_minutes: int = (cierre_aep_min - apertura_aep_min),
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

    flights: List[Avion] = []
    for i, t in enumerate(spawns):
        ac = Avion(id=i+1, momento_aparicion=t, distancia_a_aep=100.0, velocidad=300.0, status="approach")
        ac.eta_base = t + FREE_FLOW_ETA
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
        approaching.sort(key=lambda f: f.distancia_a_aep)

        # velocidad por defecto = vmax por banda (o backtrack fijo)
        leader_speed = None
        for f in approaching:
            if f.status == "backtrack":
                f.velocidad = velocidad_reversa
                continue

            vmin, vmax = f.velocidad_permitida()
            desired = vmax  # sin congestión
            # separación con el de adelante (si existe)
            if leader_speed is not None:
                # estimar gap temporal tosco: diferencia de distancias divididas por speeds (heurística)
                # (aprox rápido para decidir si hay riesgo; lo refinado implicaría predecir ETAs por integración)
                lead = leader  # definido en loop anterior
                # tiempo restante estimado (min) = d / (v_nm/min)
                t_self = f.distancia_a_aep / (desired/60.0)
                t_lead = lead.distancia_a_aep / (lead.velocidad/60.0)
                gap = (t_self - t_lead)

                if gap < separacion_minima:
                    # regla: el follower baja 20 kts vs el líder hasta lograr >=5 min
                    candidate = min(desired, max(vmin, lead.velocidad - 20.0))
                    if candidate < vmin:
                        # congestión fuerte: entra en backtrack
                        f.status = "backtrack"
                        f.velocidad = velocidad_reversa
                        f.sufrio_congestion = True
                    else:
                        f.velocidad = candidate
                        f.sufrio_congestion = True
                        # reeval gap tosco; si sigue corto, lo dejamos para el próximo minuto
                else:
                    f.velocidad = desired
            else:
                f.velocidad = desired

            leader_speed = f.velocidad
            leader = f  # para el próximo

        # avanzar todos
        for f in flights:
            prev_d = f.distancia_a_aep
            f.step(dt_min=1.0, open_runway=open_runway)

        # registrar aterrizajes con regla de separación real en la pista
        for f in flights:
            if f.status == "landed" and f.instante_aterrizaje is None:
                # aplica la separación en la pista (si está muy cerca del anterior, el aterrizaje se difiere)
                landing_time = float(t)
                if timeline_landings and landing_time - timeline_landings[-1] < separacion_minima:
                    # no puede aterrizar aún; forzamos un pequeño "hold" de 1 min
                    # (en un modelo más fino esto sería un go-around; aquí lo tratamos como espera corta)
                    f.status = "approach"
                    f.distancia_a_aep = max(0.5, f.distancia_a_aep)  # lo devolvemos levemente arriba
                else:
                    # OK, aterriza
                    f.instante_aterrizaje = landing_time
                    timeline_landings.append(landing_time)
                    last_landing_time = landing_time
                    # ¿go-around estocástico (viento)?
                    if windy and random.random() < 0.1:
                        # vuelve a 6 nm y lo marcamos como 'goaround' temporalmente
                        f.status = "goaround"
                        f.distancia_a_aep = 6.0
                        f.velocidad = 180.0
                        f.instante_aterrizaje = None  # todavía no había aterrizado
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
    landed = [f for f in flights if f.instante_aterrizaje is not None]
    diverted = [f for f in flights if f.status == "diverted"]
    congested = [f for f in flights if f.sufrio_congestion]

    delays = []
    for f in landed:
        if f.eta_base is not None:
            delays.append(max(0.0, f.instante_aterrizaje - f.eta_base))

    metrics = {
        "spawned": len(flights),
        "landed": len(landed),
        "diverted": len(diverted),
        "congestion_rate_per_flight": (len(congested) / max(1, len(flights))),
        "avg_delay_min": (sum(delays)/len(delays)) if delays else 0.0,
    }
    return SimResult(flights=flights, metrics=metrics, timeline_landings=timeline_landings)

# TP1 – Ejercicio 1: simulación Monte Carlo básica + visualización
def plot_trayectorias(flights: List[Avion]):
    """
    Dibuja distancia a AEP (nm) vs tiempo (min) para un subconjunto de aviones.
    """
    for f in flights:
        ts = [p[0] for p in f.historial]
        ds = [p[1] for p in f.historial]
        if ts:
            plt.plot(ts, ds, alpha=0.7, label=f"Avión {f.id}")

    plt.gca().invert_yaxis()  # opcional: 100 nm arriba, 0 nm abajo (visual)
    plt.xlabel("Tiempo (min desde 06:00)")
    plt.ylabel("Distancia a AEP (nm)")
    plt.title("Aproximaciones (distancia vs tiempo)")

    # fijar rango del eje x a 0–1080 min
    plt.xlim(0, 1080)

    if len(flights) <= 12:
        plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parámetros de prueba
    lam = 0.1                 # probabilidad por minuto de nuevo avión
    seed = 42                 # semilla aleatoria
    windy = False             # activar go-arounds (10%)
    cierre = None             # minuto del día en que cerrar pista (None = nunca)
    duracion_cierre = 30      # minutos de cierre

    # Correr simulación
    resultado = simulate_day(
        lam=lam,
        seed=seed,
        windy=windy,
        closure_minute=cierre,
        closure_duration=duracion_cierre
    )

    # Mostrar métricas
    print("Métricas del día:")
    for k, v in resultado.metrics.items():
        print(f"  {k}: {v}")

    # Graficar trayectorias
    plot_trayectorias(resultado.flights)
