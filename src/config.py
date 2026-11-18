"""
Constantes globales del proyecto.

Es el único lugar donde viven la semilla, las rutas y la taxonomía de intenciones, para que todos
los módulos usen los mismos valores. Cualquier cambio aquí afecta a todo el equipo, por favor avisar
en el grupo y de una vez integrar a develop con PR.
Si se quiere cambiar la taxonomía, por favor discutirlo entre todos.
"""

import os
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

# Motor M2 (búsqueda semántica sobre la base de conocimiento).
# Modelo de embeddings usado por M2 (y por el recuperador de M3). Elegido con
# ``python -m src.eval.compare_embeddings`` sobre las preguntas reales de train: mayor Hit@1
# (con M2_SPLIT_PARAGRAPHS = True). Comparación en ``reports/m2_model_selection.csv``.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Modelos que se comparan para elegir EMBEDDING_MODEL (todos gratuitos y multilingües).
EMBEDDING_CANDIDATES = (
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-small",
)
# Cuántos fragmentos devuelve el recuperador (Hit@1, Hit@3 y Hit@5 se miden con k = 5).
M2_TOP_K = 5
# Si es True, cada párrafo de un fragmento se vectoriza por separado (mejor Hit@1 en la
# comparación de modelos que un solo vector por fragmento).
M2_SPLIT_PARAGRAPHS = True

# Motor M3 (RAG con un modelo de lenguaje local en Ollama).
LLM_MODEL = "qwen3.6:35b-a3b"  # copiado EXACTO de la salida de `ollama list`
LLM_TEMPERATURE = (
    0.0  # 0 = respuestas reproducibles (la semilla SEED va en las opciones)
)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_NUM_CTX = (
    4096  # ventana de contexto pedida a Ollama (el valor por defecto es enorme)
)
LLM_MAX_TOKENS = 300  # tope de tokens generados por respuesta
LLM_TIMEOUT_S = 300.0  # segundos máximos de espera por respuesta
# Cuántos fragmentos recuperados se le entregan al modelo como contexto.
M3_CONTEXT_K = 5
# Hardware donde se mide la latencia (se reporta en el informe).
HARDWARE_LABEL = "AMD Ryzen 5 8600HS, NVIDIA RTX 4050 6 GB, 64 GB RAM"
