# Bitácora de Ingeniería (Decision Intelligence Capstone)

**Cátedra:** Estadística Aplicada y Modelos de Regresión (EAyMR 2026)  
**Institución:** Pontificia Universidad Católica Argentina (UCA) — Facultad de Ingeniería y Ciencias Agrarias  
**Proyecto:** Sistema de Inteligencia de Decisiones para Mitigación de Riesgo Crediticio y Evaluación de Políticas Financieras  
**Dataset Oficial:** HMEQ (Home Equity Loan Dataset, $N = 5.960$, 13 variables originales)

---

## Entrada 1 — Marco Estratégico, Contrato de Datos y Partición sin Data Leakage
*Responsable: Project Lead & Scrum Master*  
*Auditoría: QA & Validation Engineer*  
*Fecha: Noviembre 2026*

### 1.1 Preguntas Fundacionales de Negocio
En la gestión contemporánea de riesgo crediticio, una entidad financiera debe responder a dos desafíos complementarios antes de dividir el desarrollo en sus células técnicas:
1. **Predicción Observacional ($P(Y \mid X)$):** ¿Cuál es la probabilidad calibrada de que un solicitante entre en default (`BAD = 1`) dados sus antecedentes de buró e ingresos?
2. **Intervención y Decisión Causal ($P(Y \mid do(T = t))$):** Si la entidad interviene el sistema incentivando una política de consolidación de pasivos (`TREATMENT = 1`) en lugar de mejoras del hogar (`TREATMENT = 0`), ¿cuál es el verdadero impacto causal sobre la tasa de impago, aislando la situación financiera basal ($Z$)?

### 1.2 Selección y Fundamentación del Dataset
Se seleccionó y congeló el dataset **HMEQ (Home Equity Loan Dataset)**:
- **Volumen:** $N = 5.960$ observaciones (supera holgadamente el requisito de $N \ge 1.500$).
- **Heterogeneidad:** 13 variables que combinan atributos continuos (`LOAN`, `MORTDUE`, `VALUE`, `DEBTINC`), discretos (`DELINQ`, `DEROG`, `NINQ`, `CLNO`) y categóricos (`REASON`, `JOB`).
- **Exclusión de Toy Datasets:** Cumple la prohibición explícita de la cátedra de no utilizar datasets triviales (Iris, Titanic, Boston Housing, etc.), incorporando fricción real de datos bancarios.

### 1.3 Contrato Unificado de Variables para el Equipo Scrum
Definido en el código central `src/config.py`:
- **Target Principal ($Y$):** `BAD` (binaria, 1 = default, 0 = cumplidor). Tasa global de mora = $19.95\%$.
- **Target Continuo ($Y_{cont}$):** `LOAN` (monto del crédito solicitado).
- **Tratamiento Causal ($T$):** `TREATMENT` (binaria $0/1$), derivado de `REASON` ($1$ = `DebtCon`, $0$ = `HomeImp`).
- **Factor de Agrupación ANOVA:** `JOB` (6 ocupaciones: *Mgr, Office, Other, ProfExe, Sales, Self*).
- **Covariables / Confusores Basales ($Z$):** `MORTDUE`, `VALUE`, `YOJ`, `DEROG`, `DELINQ`, `CLAGE`, `NINQ`, `CLNO`, `DEBTINC` y `JOB`.

### 1.4 Diagnóstico de Nulos y Gobernanza de Imputación
- **Auditoría de Omisión en `DEBTINC` (21.26% de nulos):** Se estimó un modelo logit:  
  $$\text{logit}(P(\text{DEBTINC is missing})) = \beta_0 + \beta_1 \text{LOAN} + \beta_2 \text{VALUE} + \beta_3 \text{BAD}$$  
  El coeficiente de `BAD` arrojó $\beta = 2.79, z = 34.74, p < 10^{-16}$, demostrando que los clientes morosos tienen un Odds Ratio de omisión $\approx 16.3$ veces mayor.
- **Decisión Metodológica:** Se rechazó formalmente el mecanismo MCAR a favor de **MAR / MNAR**. Se prohíbe el descarte ciego de filas (`dropna()`) para evitar sesgo de selección y se añade la columna indicadora `DEBTINC_is_missing`.

### 1.5 Regla Estricta Anti-Leakage y Partición Oficial
- Los datos se dividieron **antes de cualquier imputación** en:
  - **Train:** 80% ($N = 4.768$ filas, tasa de default = $19.95\%$).
  - **Test:** 20% ($N = 1.192$ filas, tasa de default = $19.97\%$).
- Las medianas se aprendieron **únicamente sobre Train** y se transfirieron a Test.
- Los datasets finales quedaron fijados y serializados en:
  - `data/processed/train.csv`
  - `data/processed/test.csv`
  - `data/processed/metadata.json`

---
