import os

# Directorios principales
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE = os.path.join(ROOT_DIR, 'BDD_ENE_17_ABR_24.csv')

# Directorio interno de la app (para guardar los precomputados)
APP_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FORECAST_RESULTS_FILE = os.path.join(APP_DATA_DIR, 'forecast_results.csv')
FORECAST_METRICS_FILE = os.path.join(APP_DATA_DIR, 'forecast_metrics.csv')

# Asegurar que la carpeta data exista
os.makedirs(APP_DATA_DIR, exist_ok=True)

# Parámetros por defecto para DBSCAN
DEFAULT_EPS = 0.5  # km
DEFAULT_MIN_SAMPLES = 5

# Colores y estilo para UI
COLOR_BG = '#0D1117'
COLOR_ACCENT = '#F97316'
COLOR_INFO = '#3B82F6'
