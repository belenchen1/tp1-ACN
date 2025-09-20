
#Helpers.py
from typing import Tuple

from Constants import VELOCIDADES, OBJ_SEP_BASE

def knots_to_nm_per_min(k: float) -> float:
    ''' convierte de nudos a nm/min (ambas son unidades de velocidad)'''
    return k / 60.0

def velocidad_por_distancia(d_nm: float) -> Tuple[float, float]:
    for dist_low, dist_high, vmin, vmax in VELOCIDADES:
        if dist_low <= d_nm < dist_high:
            return vmin, vmax # devuelve las velocidades para esa banda
    return VELOCIDADES[0][2], VELOCIDADES[0][3] # si te pasas de los 100nm

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



#Calcula el tiempo de llegada desde su aparicion hasta aterrizar en el caso que no hay congestion
#es decir a vmax por banda
#def baseline_time_from_100nm(step_nm: float = 0.1) -> float:
#    d = 100.0
#    t = 0.0
#    while d > 0:
#        vmin, vmax = velocidad_por_distancia(d)
#        v_nm_min = knots_to_nm_per_min(vmax)
#        ds = min(step_nm, d)
#        t += ds / v_nm_min
#        d -= ds
#    return t



# Helpers.py
from Constants import MINUTE
from Helpers import velocidad_por_distancia, knots_to_nm_per_min  # si esto ya está en este archivo omití el segundo import

def baseline_time_from_100nm() -> float:
    """
    Baseline alineado con la simulación:
    - cada paso es 1 MINUTE
    - se usa la banda (vmax) según la distancia al *comienzo* del minuto
    - en el último minuto se interpola la fracción
    """
    d = 100.0
    t = 0.0
    while d > 0.0:
        _, vmax = velocidad_por_distancia(d)
        v_nm_min = knots_to_nm_per_min(vmax)  # nm por minuto
        adv = v_nm_min * MINUTE

        if d - adv <= 0.0:
            # fracción del minuto que realmente hace falta
            t += d / v_nm_min
            break
        else:
            d -= adv
            t += MINUTE
    return t

# Usá este como baseline único
BASELINE_TIME_MIN = baseline_time_from_100nm()


def g_objetivo(d_nm: float) -> float:
    """
    Target de separación deseada en minutos según distancia.
    - Más lejos pedimos un poquito más para evitar compresiones aguas abajo.
    - En tramo final, mantenemos 5 min (separación mínima operacional).
    """
    if d_nm >= 50.0:
        return OBJ_SEP_BASE + 1.0   # p.ej. 7 si base=6
    if d_nm >= 15.0:
        return OBJ_SEP_BASE         # p.ej. 6
    return 4.0                      # tramo final 5 min



