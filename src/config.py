"""
src/config.py
Contrato Único de Datos y Variables para el Equipo Scrum (EAyMR 2026)
Todos los integrantes del equipo deben importar las variables desde este archivo.
"""

from pathlib import Path

# Rutas del Proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = BASE_DIR / "data" / "hmeq.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test.csv"
METADATA_PATH = DATA_PROCESSED_DIR / "metadata.json"

# Hiperparámetros de Partición
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ==============================================================================
# CONTRATO FORMAL DE VARIABLES
# ==============================================================================

# 1. Variable Objetivo (Target Y para Clasificación y Causalidad)
TARGET_BINARY = "BAD"  # 1 = Default / Cesación de Pagos, 0 = Cumplidor

# 2. Variable Objetivo Continua (para Regresión MCO en Fase 2)
TARGET_CONTINUOUS = "LOAN"  # Monto del crédito solicitado

# 3. Variable de Intervención / Política Causal (Tratamiento T para Fase 4 y App)
# Política: Otorgar crédito orientado a Consolidación de Deuda (1) vs. Mejoras del Hogar (0)
TREATMENT_VAR = "TREATMENT"  # 1 = DebtCon, 0 = HomeImp
ORIGINAL_TREATMENT_COL = "REASON"

# 4. Factor de Agrupación para ANOVA (Fase 1)
ANOVA_FACTOR = "JOB"  # Categorías ocupacionales (Mgr, Office, Other, ProfExe, Sales, Self)
ANOVA_VARIABLE = "LOAN"  # Variable continua testeada entre grupos ocupacionales

# 5. Covariables Numéricas Basales (Z)
NUMERICAL_FEATURES = [
    "LOAN",
    "MORTDUE",
    "VALUE",
    "YOJ",
    "DEROG",
    "DELINQ",
    "CLAGE",
    "NINQ",
    "CLNO",
    "DEBTINC"
]

# 6. Covariables Categóricas
CATEGORICAL_FEATURES = ["JOB"]

# 7. Confusores Basales en Inferencia Causal (Z)
CAUSAL_CONFOUNDERS = [
    "LOAN",
    "MORTDUE",
    "VALUE",
    "YOJ",
    "DEROG",
    "DELINQ",
    "CLAGE",
    "NINQ",
    "CLNO",
    "DEBTINC"
]
