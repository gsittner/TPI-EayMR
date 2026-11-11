# Bitácora de Ingeniería (Decision Intelligence Capstone)

**Cátedra:** Estadística Aplicada y Modelos de Regresión (EAyMR 2026)  
**Institución:** Pontificia Universidad Católica Argentina (UCA) — Facultad de Ingeniería y Ciencias Agrarias  
**Proyecto:** Sistema de Inteligencia de Decisiones para Mitigación de Riesgo Crediticio y Evaluación de Políticas Financieras  
**Dataset Oficial:** HMEQ (Home Equity Loan Dataset, $N = 5.960$, 13 variables)

---

## Entrada 1 — Marco Estratégico e Hipótesis Fundacionales
*Responsable: Project Lead & Scrum Master*  
*Fecha: Noviembre 2026*

### 1.1 Preguntas Fundacionales de Negocio
En la gestión contemporánea de riesgo de crédito, una entidad financiera no solo busca clasificar qué solicitantes tienen alta probabilidad observacional de entrar en morosidad ($P(Y=1 \mid X)$), sino evaluar el impacto de sus políticas crediticias:
1. **Predicción Observacional:** Dado el perfil financiero y laboral del solicitante ($X$), ¿cuál es la probabilidad calibrada de que entre en cesación de pagos ($Y = \text{BAD}$)?
2. **Intervención y Decisión Causal:** Si el comité de riesgo incentiva o reorienta la solicitud hacia una política de consolidación de pasivos frente a mejoras edilicias ($do(T = \text{DebtCon})$), ¿cuál es el verdadero impacto causal sobre el impago ($Y$), aislando los factores de confusión basales ($Z$)?

### 1.2 Selección y Justificación del Dataset
Se seleccionó el dataset **HMEQ (Home Equity)** que contiene información de 5.960 créditos hipotecarios y de liquidez con garantía inmobiliaria.
- **Volumen:** $N = 5.960 > 1.500$ (cumple holgadamente el umbral mínimo).
- **Dimensionalidad:** 13 variables (mezcla de continuas como `LOAN`, `MORTDUE`, `VALUE`, `DEBTINC`, discretas como `DELINQ`, `DEROG`, `CLNO` y categóricas como `REASON` y `JOB`).
- **Fricción del dato real:** Presenta un 21.2% de datos faltantes no aleatorios en el ratio deuda/ingreso (`DEBTINC`), asimetría severa en montos de crédito y colas pesadas, lo que descarta el sesgo de "toy datasets" prohibidos por la cátedra.

---

## Entrada 2 — Matriz de Decisiones Estadísticas y Supuestos (Fase 1)
*Responsable: Backend & Machine Learning / Causal Engineer*  
*Auditoría: QA & Validation Engineer*  
*Fecha: Noviembre 2026*

### 2.1 Diagnóstico del Mecanismo de Pérdida de Datos (Missing Data)
- **Variable auditada:** `DEBTINC` (ratio deuda/ingreso), con 1.267 valores faltantes (21.26%).
- **Hipótesis testeadas:**
  - $H_0$ (MCAR): La probabilidad de omisión en `DEBTINC` es independiente de las variables observadas.
  - $H_1$ (MAR / MNAR): La probabilidad de omisión depende de covariables observadas.
- **Resultado econométrico:** Se ajustó una regresión logística modelando la probabilidad de ausencia:
  $$\text{logit}(P(\text{DEBTINC is missing})) = \beta_0 + \beta_1 \text{LOAN} + \beta_2 \text{VALUE} + \beta_3 \text{BAD}$$
  El coeficiente de $\text{BAD}$ resultó positivo y abrumadoramente significativo ($\beta = 2.84, z = 36.26, p < 10^{-16}$). Los clientes en morosidad tienen un Odds Ratio de omisión $\approx 17.1$ veces mayor.
- **Decisión de remediación:** Se rechaza de forma determinística la hipótesis MCAR. La pérdida es como mínimo **MAR (Missing at Random)** con fuerte sospecha de **MNAR (Missing Not at Random)** (solicitantes en dificultades financieras omiten o no declaran sus ingresos totales). En la preparación de datos se prohíbe el descarte ciego de filas (*listwise deletion*) para evitar sesgo de selección severo; se opta por imputación condicional con indicador dummy de ausencia (*missingness indicator*).

### 2.2 Diagnóstico de Supuestos para ANOVA y Contrastes
- **Normalidad en Grandes Muestras ($N = 5.960$):**
  - El test de Shapiro-Wilk arrojó $p < 10^{-20}$ para todos los subgrupos de `JOB`.
  - **Decisión metodológica:** Siguiendo la advertencia de la cátedra, ante muestras grandes las desviaciones infinitesimales de la campana de Gauss invalidan el test de Shapiro-Wilk. Se evaluó la asimetría empírica ($\text{skewness} \in [-1, 1]$) y curtosis. Debido a la asimetría positiva en `LOAN`, se respalda el Teorema Central del Límite para las medias muestrales, pero se contrastan los resultados con pruebas no paramétricas (Mann-Whitney $U$ y Kruskal-Wallis).
- **Homocedasticidad (Test de Levene):**
  - Al evaluar la varianza de `LOAN` a través de las categorías de `JOB`, el test de Levene arrojó $W = 25.08, p = 4.32 \times 10^{-25}$.
  - **Decisión de remediación:** Dado el rechazo contundente de la homogeneidad de varianzas, el ANOVA tradicional de Fisher queda invalidado formalmente. Se adopta la estadística $F^*$ de **Welch's ANOVA** ($F_{\text{Welch}} = 29.84, p < 10^{-20}$) y comparaciones post-hoc robustas (Games-Howell / Bonferroni).

### 2.3 Resumen de Contrastes de Hipótesis Formales
| Contraste | Hipótesis $H_0$ | Hipótesis $H_1$ | Estadístico | $p$-valor exacto | Tamaño de Efecto (Cohen's $d$) | IC 95% Diferencia | Conclusión de Negocio |
|---|---|---|---|---|---|---|---|
| **Contraste 1 (LOAN por BAD)** | $\mu_{\text{cumplidor}} = \mu_{\text{moroso}}$ | $\mu_{\text{cumplidor}} \neq \mu_{\text{moroso}}$ | Welch $t = 5.720$ (MWU $z = 6.45$) | $1.25 \times 10^{-8}$ | $d = -0.187$ (Pequeño) | $[-\$2.714, -\$1.312]$ | Los créditos en default tuvieron montos promedio menores (\$16.966 vs \$18.979), reflejo del racionamiento crediticio previo del banco hacia perfiles riesgosos. |
| **Contraste 2 (DEBTINC por REASON)** | $\mu_{\text{DebtCon}} = \mu_{\text{HomeImp}}$ | $\mu_{\text{DebtCon}} > \mu_{\text{HomeImp}}$ | Welch $t = 3.795$ (MWU $z = 4.02$) | $1.51 \times 10^{-4}$ | $d = 0.124$ (Pequeño) | $[+0.52\%, +1.61\%]$ | Quienes solicitaron crédito para consolidar deudas presentan un ratio deuda/ingreso promedio significativamente mayor que quienes lo pidieron para reformas hogareñas. |

---


## Entrada 3 — Autopsia de Experimentos Fallidos (Data Autopsy) y Matriz de Supuestos (Fase 2)
*Responsable: Backend & Machine Learning / Causal Engineer*  
*Auditoría: QA & Validation Engineer*  
*Fecha: Noviembre 2026*

### 3.1 Auditoría Econométrica de Gauss-Markov en el Modelo de Referencia
Se estimó el modelo clásico de referencia MCO:
$$\text{LOAN} = \beta_0 + \beta_1 \text{MORTDUE} + \beta_2 \text{VALUE} + \beta_3 \text{YOJ} + \beta_4 \text{DEROG} + \beta_5 \text{DELINQ} + \beta_6 \text{CLAGE} + \beta_7 \text{NINQ} + \beta_8 \text{CLNO} + \beta_9 \text{DEBTINC} + \varepsilon$$

- **Bondad de Ajuste Global:** $R^2 = 0.181, R^2_{\text{adj}} = 0.179, F(9, 3505) = 86.20, p < 10^{-140}$.
- **Auditoría de Supuestos:**
  1. **Homocedasticidad (Breusch-Pagan):** $LM = 46.75, p = 4.37 \times 10^{-7}$. Se rechaza de forma categórica la varianza residual constante. Los préstamos de mayor envergadura exhiben una dispersión de residuos marcadamente heterogénea (cono de dispersión).
  2. **Normalidad Residual (Jarque-Bera):** $JB = 20.097, p < 10^{-50}$, con asimetría positiva ($\text{Skew} = 2.50$) y curtosis leptocúrtica severa ($\text{Kurt} = 13.60$).
  3. **Independencia Residual (Durbin-Watson):** En el orden de carga original del dataset, $DW = 0.325$, revelando que los registros se encontraban ordenados en bloques de créditos de menor a mayor monto. Al aleatorizar el índice, $DW \approx 1.98$, confirmando ausencia de dependencia estructural temporal genuina.

### 3.2 Autopsia de Experimento Fallido #1: Multicolinealidad Estructural entre Colaterales Inmobiliarios
- **Hipótesis inicial:** Incluir simultáneamente `VALUE` (valor tasado de la propiedad) y `MORTDUE` (saldo hipotecario pendiente) para capturar el apalancamiento bruto del cliente.
- **Falla detectada:** Ambas variables exhiben una correlación de Pearson $\rho = 0.87$. En el modelo MCO sin regularizar, sus Factores de Inflación de Varianza se elevan a $\text{VIF} > 4.4$, y al descomponer sin centrar superan $\text{VIF} > 10$. Esto generó inestabilidad en los errores estándar y un signo contraintuitivo negativo en `MORTDUE` ($\beta = -0.077, t = -9.49$) al actuar como supresor de `VALUE`.
- **Remediación adoptada:**
  1. En PCA se agrupan en una única dimensión de *Colateral Inmobiliario y Solvencia Patrimonial* (PC1).
  2. En la Fase 3 se incorporará penalización $L_2$ (Ridge / ElasticNet) para encoger los coeficientes correlacionados sin inflar la varianza de los estimadores.

### 3.3 Autopsia de Experimento Fallido #2: Sensibilidad Extrema a Observaciones de Alto Apalancamiento (*Leverage*)
- **Falla detectada:** El análisis de influencia identificó 283 observaciones con apalancamiento superior al umbral crítico $h_{ii} > \frac{2(p+1)}{n} = 0.00569$, y 185 observaciones con Distancia de Cook $D_i > \frac{4}{n} = 0.00114$. Estos casos corresponden a solicitantes con propiedades de lujo ($> \$500.000$) y préstamos atípicos que traccionan desproporcionadamente la recta de regresión por MCO.
- **Remediación adoptada:** La exclusión arbitraria de estas filas distorsionaría el portafolio real del banco. Por tanto, se calibró una **Regresión Robusta (Huber / RANSAC)** con función de pérdida bicuadrada de Tukey, logrando estimadores inmunes al tironeo de las colas pesadas.

### 3.4 Reducción de Dimensionalidad con PCA libre de Fuga de Datos (*Data Leakage*)
- **Procedimiento determinístico:** Se particionó la muestra en 80% Train y 20% Test. Se computó la estandarización ($\mu, \sigma$) y la matriz de covarianza / descomposición espectral **exclusivamente sobre Train**. Los datos de Test fueron proyectados ciegamente con los autovectores aprendidos.
- **Criterio de Kaiser ($\lambda_k > 1$):** Retiene exactamente **3 componentes principales**:
  - $\text{PC1}$ ($\lambda_1 = 2.14$, 23.78% varianza): Dimensión de *Capacidad Patrimonial y Colateral Inmobiliario* (altas cargas positivas en `MORTDUE` y `VALUE`).
  - $\text{PC2}$ ($\lambda_2 = 1.43$, 15.90% varianza): Dimensión de *Riesgo Moroso e Historial Negativo* (cargas dominantes en `DELINQ`, `DEROG` y consultas recientes `NINQ`).
  - $\text{PC3}$ ($\lambda_3 = 1.21$, 13.42% varianza): Dimensión de *Madurez Crediticia y Estabilidad Laboral* (cargas dominantes en `CLAGE` y `YOJ`).
- **Varianza acumulada:** Las 3 primeras componentes capturan el **53.1%** de la varianza total multivariada libre de colinealidad.

---
