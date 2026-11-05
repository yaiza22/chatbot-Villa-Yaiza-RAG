"""
Constantes globales del proyecto.

Es el único lugar donde viven la semilla, las rutas y la taxonomía de intenciones, para que todos
los módulos usen los mismos valores. Cualquier cambio aquí afecta a todo el equipo, por favor avisar
en el grupo y de una vez integrar a develop con PR.
Si se quiere cambiar la taxonomía, por favor discutirlo entre todos.
"""

from pathlib import Path

SEED = 42

# Rutas (siempre relativas al repo).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = DATA_DIR / "kb"
REPORTS_DIR = REPO_ROOT / "reports"

# Taxonomía de intenciones.
INTENTS = (
    "precio",
    "capacidad",
    "reglas_servicios",
    "ubicacion",
    "disponibilidad",
    "otro_negocio",
    "fuera_de_alcance",
)

# Intenciones en las que el bot debe escalar a una persona o abstenerse.
ESCALATING_INTENTS = frozenset(
    {"precio", "disponibilidad", "otro_negocio", "fuera_de_alcance"}
)

# Partición de los conjuntos etiquetados.
TEST_FRACTION = 0.3
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
