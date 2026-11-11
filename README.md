# Decision Intelligence Capstone: Gobernanza de Datos y Contrato del Equipo

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Methodology Scrum](https://img.shields.io/badge/methodology-Scrum-orange.svg)](#organización-del-equipo)
[![Dataset HMEQ](https://img.shields.io/badge/dataset-HMEQ%20(5.960%20filas)-blueviolet.svg)](#-contrato-formal-de-variables)

Trabajo Práctico Integrador para la cátedra **Estadística Aplicada y Modelos de Regresión (EAyMR 2026)** — Pontificia Universidad Católica Argentina (UCA).

---

## 📌 Propósito de esta Entrega / Hito Inicial (Data Foundation)

Este repositorio establece la **base canónica y congelada de datos** para que los integrantes del equipo Scrum desarrollen sus respectivas secciones de forma paralela y sin discrepancias:
1. **Selección y descarga del dataset oficial HMEQ** ($N = 5.960$, no es *toy dataset*).
2. **Definición del Contrato Formal de Variables** ($T, Y, Z$, factor ANOVA).
3. **Diagnóstico del mecanismo de pérdida de datos** (rechazo formal de MCAR a favor de MAR/MNAR en `DEBTINC`).
4. **Limpieza e Imputación sin Fuga de Datos (*No Data Leakage*)**: medianas aprendidas estrictamente en Train.
5. **Partición oficial estratificada**: `data/processed/train.csv` (80%) y `data/processed/test.csv` (20%).

---

## 🏛 Estructura Modular del Proyecto

```text
TPI/
├── data/
│   ├── hmeq.csv                 # Dataset original crudo (N = 5.960)
│   └── processed/               # Datasets limpios y particionados para el equipo
│       ├── train.csv            # Partición Train (80%, N = 4.768)
│       ├── test.csv             # Partición Test  (20%, N = 1.192)
│       └── metadata.json        # Metadatos y medianas aprendidas en Train
├── notebooks/
│   └── TP_Integrador_Final.ipynb # Notebook maestro reproducible con visualizaciones
├── src/
│   ├── __init__.py
│   ├── config.py                # Contrato formal de variables único del equipo
│   └── data_pipeline.py         # Pipeline de limpieza, imputación y partición
├── tests/
│   └── test_data_pipeline.py    # Quality Gate con pytest (5/5 tests pasando)
├── BITACORA.md                  # Entrada 1: Marco Estratégico y Decisiones de Datos
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación y guía de reproducción
```

---

## 🎯 Contrato Formal de Variables (`src/config.py`)

Todos los integrantes del equipo deben importar las variables directamente desde `src/config.py`:

| Rol Metodológico | Variable en Dataset | Tipo | Definición Conceptual de Negocio |
|---|---|---|---|
| **Target Primario ($Y$)** | `BAD` | Binaria ($0, 1$) | $1$ = Cesación de pagos / Default, $0$ = Cumplidor. Tasa global: $19.95\%$. |
| **Target Secundario ($Y_{cont}$)** | `LOAN` | Continua (\$) | Monto del préstamo solicitado. |
| **Tratamiento Causal ($T$)** | `TREATMENT` | Binaria ($0, 1$) | $1$ = Consolidación de Deuda (`DebtCon`), $0$ = Reformas del Hogar (`HomeImp`). |
| **Factor ANOVA ($k \ge 3$)** | `JOB` | Categórica (6 niveles) | Ocupación profesional: *Mgr, Office, Other, ProfExe, Sales, Self*. |
| **Covariables Basales ($Z$)** | `MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC` | Mixtas | Solvencia financiera, colateral hipotecario y buró crediticio. |

---

## 🚀 Guía Rápida para el Equipo

### 1. Clonar el Repositorio e Instalar Dependencias
```bash
git clone <URL_DEL_REPOSITORIO>
cd TPI
pip install -r requirements.txt
```

### 2. Ejecutar la Puerta de Calidad (Quality Gate)
```bash
python -m pytest tests/ -v
```

### 3. Usar los Datos Procesados en tu Sección
```python
import pandas as pd
from src.config import TARGET_BINARY, TREATMENT_VAR, NUMERICAL_FEATURES

train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
```
