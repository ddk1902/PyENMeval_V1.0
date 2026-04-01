from pathlib import Path

# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# carpetas principales
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "tmp"

# crear si no existen
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)