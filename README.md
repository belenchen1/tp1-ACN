# TP1 AEP — Starter Package

Este paquete te da un **andamiaje** para simular el problema de aproximaciones a AEP.
Incluye:
- `aep_sim_starter.py`: motor de simulación en tiempo discreto (Δt=1 min) con las reglas base.
- `experiments_demo.py`: ejemplos reproducibles para correr los ítems (2)–(4) y bosquejos para (5)–(7).
- `requirements.txt`: solo `matplotlib` y `numpy` (puedes quitarlo si no los usas).
- `LICENSE`: MIT (pueden modificar libremente).

> Nota: El comportamiento de “virar 180° y buscar gap ≥10 min” está bosquejado como **TODO** claramente marcado.
  El flujo base ya calcula congestión, demoras y desvíos a Montevideo (cuando no encuentra gap).

## Cómo usar

```bash
python experiments_demo.py --lam 0.1 --hours 100
```

- Cambiá `--windy` para activar el 10% de go-arounds (ítem 5).
- Usá `--closure "12:00"` para simular un cierre sorpresivo de 30 min (ítem 6).
- El script guarda métricas y gráficos en la carpeta `./out`.

## ¿Qué falta para el TP completo?
- Completar la lógica de **reingreso tras backtrack** al detectar el primer hueco ≥10 min (TODO etiquetado).
- Agregar **intervalos de confianza** a todas las métricas (plantillas incluidas).
- Terminar la **visualización por timeline** (se incluye un primer gráfico).

