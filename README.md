# Decision Intelligence Capstone: Modelado Predictivo, Inferencia Causal y Sistemas de Decisión en Producción

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Architecture Scrum](https://img.shields.io/badge/methodology-Scrum-orange.svg)](#organización-del-equipo)
[![Causal ML](https://img.shields.io/badge/causal_ml-DoWhy%20%26%20EconML-purple.svg)](#fase-4-inferencia-causal)

Trabajo Práctico Integrador para la asignatura **Estadística Aplicada y Modelos de Regresión (EAyMR 2026)** — Pontificia Universidad Católica Argentina (UCA).

---

## 🏛 Arquitectura y Estructura del Repositorio

```text
TPI/
├── data/                       # Ingesta y almacenamiento del dataset HMEQ
│   └── hmeq.csv
├── notebooks/                  # Notebook maestro reproducible de 4 fases
│   └── TP_Integrador_Final.ipynb
├── src/                        # Código modular de ingeniería y modelos
│   ├── __init__.py
│   ├── phase1_inference.py     # Inferencia clásica, ANOVA y MCAR/MAR
│   └── build_notebook.py       # Pipeline generador de notebook
├── tests/                      # Suite determinística de calidad (Quality Gate)
│   └── test_phase1.py
├── app/                        # Aplicación interactiva de toma de decisiones
├── BITACORA.md                 # Cuaderno de laboratorio con las 5 entradas
├── requirements.txt            # Dependencias fijadas del proyecto
└── README.md                   # Documentación técnica y guía de reproducción
```

---

## 👥 Organización del Equipo bajo Marco Ágil (Scrum)
1. **Project Lead & Scrum Master (Product Owner):** Arquitectura modular, trazabilidad de historias de usuario y Quality Gate.
2. **Backend & Machine Learning / Causal Engineer:** Pipelines estadísticos, econometría de Gauss-Markov, Torneo ML y estimación causal con DoWhy / EconML.
3. **Frontend & UI/UX Engineer:** Aplicación web en Streamlit con módulos "What-Is" (predictivo) y "What-If" (intervención causal).
4. **QA & Validation Engineer:** Auditoría de no fuga de datos (*data leakage*), pruebas determinísticas automáticas (`pytest`) y verificación de reproducibilidad.

---

## 🚀 Guía de Reproducción

### 1. Clonación e Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecución de Pruebas de Calidad (Quality Gate)
```bash
python -m pytest tests/ -v
```

### 3. Ejecución del Notebook Maestro
Abrir y ejecutar de punta a punta:
`notebooks/TP_Integrador_Final.ipynb`
