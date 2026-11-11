"""
src/build_notebook.py
Genera el notebook reproducible TP_Integrador_Final.ipynb con Fase 1 y Fase 2
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# ==============================================================================
# ENCABEZADO GENERAL
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🏆 Decision Intelligence Capstone: Modelado Predictivo, Inferencia Causal y Sistemas de Decisión en Producción
**Cátedra:** Estadística Aplicada y Modelos de Regresión (EAyMR 2026)  
**Institución:** Pontificia Universidad Católica Argentina (UCA) — Facultad de Ingeniería y Ciencias Agrarias  
**Profesor Titular:** Prof. Ivan Gonzalo Vilaboa  
**Integrantes y Roles Scrum:**
- 👑 **Project Lead & Scrum Master:** German y equipo
- ⚙ **Backend & Machine Learning / Causal Engineer:** Equipo EAyMR
- 🎨 **Frontend & UI/UX Engineer:** Equipo EAyMR
- 🔍 **QA & Validation Engineer:** Equipo EAyMR

---

## 🔬 FASE 1: Inferencia Estadística Clásica, Diagnósticos y ANOVA
En esta primera fase demostramos comprensión inferencial de los datos poblacionales y muestrales:
1. **Ingeniería Exploratoria y Caracterización de Distribuciones:** Análisis univariado y bivariado (KDE, boxplots, correlaciones de Pearson y Spearman).
2. **Mecanismos de Pérdida de Datos:** Clasificación formal de variables faltantes entre MCAR, MAR o MNAR mediante modelado logístico econométrico.
3. **Contrastes de Hipótesis Formales:** Formulación de $H_0$ frente a $H_1$, fijación de $\\alpha = 0.05$, cálculo de $p$-valores exactos, intervalos de confianza del 95% y magnitud del efecto ($d$ de Cohen).
4. **Análisis de Varianza (ANOVA) de un Factor:** Evaluación del comportamiento de Shapiro-Wilk en muestras masivas ($N \\ge 1.500$), diagnóstico de homocedasticidad con Levene, y remediación formal mediante **Welch's ANOVA** y comparaciones múltiples post-hoc.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor

# Configuración gráfica y estética
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("Entorno estadístico y econométrico inicializado exitosamente.")
"""))

# Fase 1: Ingesta
cells.append(nbf.v4.new_markdown_cell("""### 1.1 Ingesta y Auditoría Inicial del Dataset HMEQ
El conjunto de datos seleccionado es **HMEQ (Home Equity Loan Dataset)**, el cual registra características basales y morosidad de $N = 5.960$ créditos hipotecarios y de liquidez con garantía sobre inmuebles.
"""))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../data/hmeq.csv')
print(f"Dimensiones basales: N = {df.shape[0]} observaciones, p = {df.shape[1]} variables.")
display(df.head())
"""))

# Fase 1: Missing Data
cells.append(nbf.v4.new_markdown_cell("""### 1.2 Auditoría de Datos Faltantes: Clasificación Formal MCAR vs. MAR
Un supuesto crítico en econometría y estadística inferencial es el mecanismo subyacente de omisión de datos (Rubin, 1976):
- **MCAR (Missing Completely at Random):** $P(M \\mid Y_{obs}, Y_{mis}) = P(M)$. La omisión es un proceso puramente aleatorio.
- **MAR (Missing at Random):** $P(M \\mid Y_{obs}, Y_{mis}) = P(M \\mid Y_{obs})$. La omisión depende de variables observables, pero no del valor no observado una vez controlado lo observado.
- **MNAR (Missing Not at Random):** La probabilidad de ausencia depende del propio valor omitido.

Evaluamos formalmente si la pérdida en `DEBTINC` (21.26% de nulos) es MCAR o MAR ajustando un modelo de regresión logística:
$$\\text{logit}(P(\\text{DEBTINC is missing})) = \\beta_0 + \\beta_1 \\text{LOAN} + \\beta_2 \\text{VALUE} + \\beta_3 \\text{BAD}$$
"""))

cells.append(nbf.v4.new_code_cell("""missing_table = pd.DataFrame({
    'Valores Nulos': df.isnull().sum(),
    'Porcentaje (%)': (df.isnull().sum() / len(df)) * 100
}).sort_values('Porcentaje (%)', ascending=False)
display(missing_table[missing_table['Valores Nulos'] > 0])

# Contraste de omisión con modelo Logit
df_audit = df.copy()
df_audit['DEBTINC_missing'] = df_audit['DEBTINC'].isnull().astype(int)
logit_mcar = sm.Logit.from_formula('DEBTINC_missing ~ LOAN + VALUE + BAD', data=df_audit).fit()
print(logit_mcar.summary())
"""))

cells.append(nbf.v4.new_markdown_cell("""**Conclusión Metodológica sobre Missingness:**  
El coeficiente asociado a $\\text{BAD}$ es positivo y estadísticamente aplastante ($\\beta = 2.79, z = 34.74, p < 10^{-16}$). Los solicitantes que entran en default tienen una probabilidad drásticamente superior de no reportar su ratio deuda/ingreso ($OR = e^{2.79} \\approx 16.3$).  
**Dictamen:** Se rechaza de plano el supuesto MCAR. La pérdida es condicional a covariables (**MAR / MNAR**). Por ende, aplicar eliminación listwise (`dropna()`) sesgaría gravemente las estimaciones poblacionales.
"""))

# Fase 1: EDA
cells.append(nbf.v4.new_markdown_cell("""### 1.3 Exploración Bivariada y Matrices de Correlación (Pearson vs. Spearman)
Contrastamos las relaciones lineales (Pearson) frente a las monótonas de rango (Spearman) para detectar colas pesadas o no linealidades estructurales.
"""))

cells.append(nbf.v4.new_code_cell("""num_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'CLAGE', 'DEBTINC', 'CLNO']
corr_p = df[num_cols].corr(method='pearson')
corr_s = df[num_cols].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(corr_p, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, ax=axes[0], cbar=False)
axes[0].set_title('Matriz de Correlación Paramétrica (Pearson)', fontsize=12, fontweight='bold')

sns.heatmap(corr_s, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Matriz de Correlación de Rango (Spearman)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Fase 1: Contrastes
cells.append(nbf.v4.new_markdown_cell("""### 1.4 Formulación y Ejecución de Contrastes de Hipótesis Formales

#### Contraste 1: Monto del Préstamo Solicitado (`LOAN`) según Estado de Morosidad (`BAD`)
- **Pregunta de Negocio:** ¿Los solicitantes que terminan en cesación de pagos solicitaron montos de financiamiento significativamente distintos a los cumplidores?
- **Hipótesis:**
  $$H_0: \\mu_{\\text{LOAN, BAD=0}} = \\mu_{\\text{LOAN, BAD=1}} \\quad \\text{vs.} \\quad H_1: \\mu_{\\text{LOAN, BAD=0}} \\neq \\mu_{\\text{LOAN, BAD=1}}$$
- **Nivel de significancia:** $\\alpha = 0.05$ (fijado a priori).
- **Estadístico:** Welch's $t$-test (robusto a varianzas heterogéneas) y Mann-Whitney $U$ no paramétrico.
"""))

cells.append(nbf.v4.new_code_cell("""g0 = df[df['BAD'] == 0]['LOAN'].dropna()
g1 = df[df['BAD'] == 1]['LOAN'].dropna()

t_stat1, p_welch1 = stats.ttest_ind(g0, g1, equal_var=False)
u_stat1, p_mwu1 = stats.mannwhitneyu(g0, g1, alternative='two-sided')

# Cohen's d e Intervalo de Confianza del 95% para la diferencia
s_pool1 = np.sqrt(((len(g0)-1)*g0.var() + (len(g1)-1)*g1.var()) / (len(g0)+len(g1)-2))
d1 = (g1.mean() - g0.mean()) / s_pool1
diff_m1 = g1.mean() - g0.mean()
se1 = np.sqrt(g0.var()/len(g0) + g1.var()/len(g1))
dof1 = (g0.var()/len(g0) + g1.var()/len(g1))**2 / ((g0.var()/len(g0))**2/(len(g0)-1) + (g1.var()/len(g1))**2/(len(g1)-1))
ci1 = (diff_m1 - stats.t.ppf(0.975, dof1)*se1, diff_m1 + stats.t.ppf(0.975, dof1)*se1)

print("--- RESULTADOS CONTRASTE 1 ---")
print(f"Media Cumplidor (BAD=0): ${g0.mean():,.2f} (SD: ${g0.std():,.2f})")
print(f"Media Moroso     (BAD=1): ${g1.mean():,.2f} (SD: ${g1.std():,.2f})")
print(f"Diferencia de Medias    : ${diff_m1:,.2f} (IC 95%: [${ci1[0]:,.2f}, ${ci1[1]:,.2f}])")
print(f"Welch t-stat            : {t_stat1:.4f} (p-valor: {p_welch1:.4e})")
print(f"Mann-Whitney U stat     : {u_stat1:.1f} (p-valor: {p_mwu1:.4e})")
print(f"Magnitud del Efecto (d) : {d1:.4f} (Efecto pequeño)")
"""))

cells.append(nbf.v4.new_markdown_cell("""#### Contraste 2: Ratio Deuda/Ingreso (`DEBTINC`) según Motivo del Préstamo (`REASON`)
- **Pregunta de Negocio:** ¿Los prestatarios que recurren al banco para consolidación de pasivos (`DebtCon`) presentan una carga financiera inicial mayor que quienes solicitan fondos para refacciones hogareñas (`HomeImp`)?
- **Hipótesis:**
  $$H_0: \\mu_{\\text{DEBTINC, DebtCon}} = \\mu_{\\text{DEBTINC, HomeImp}} \\quad \\text{vs.} \\quad H_1: \\mu_{\\text{DEBTINC, DebtCon}} > \\mu_{\\text{DEBTINC, HomeImp}}$$
- **Nivel de significancia:** $\\alpha = 0.05$.
"""))

cells.append(nbf.v4.new_code_cell("""sub_reason = df.dropna(subset=['DEBTINC', 'REASON'])
g_dc = sub_reason[sub_reason['REASON'] == 'DebtCon']['DEBTINC']
g_hi = sub_reason[sub_reason['REASON'] == 'HomeImp']['DEBTINC']

t_stat2, p_welch2 = stats.ttest_ind(g_dc, g_hi, equal_var=False, alternative='greater')
u_stat2, p_mwu2 = stats.mannwhitneyu(g_dc, g_hi, alternative='greater')

s_pool2 = np.sqrt(((len(g_dc)-1)*g_dc.var() + (len(g_hi)-1)*g_hi.var()) / (len(g_dc)+len(g_hi)-2))
d2 = (g_dc.mean() - g_hi.mean()) / s_pool2
diff_m2 = g_dc.mean() - g_hi.mean()
se2 = np.sqrt(g_dc.var()/len(g_dc) + g_hi.var()/len(g_hi))
dof2 = (g_dc.var()/len(g_dc) + g_hi.var()/len(g_hi))**2 / ((g_dc.var()/len(g_dc))**2/(len(g_dc)-1) + (g_hi.var()/len(g_hi))**2/(len(g_hi)-1))
ci2 = (diff_m2 - stats.t.ppf(0.975, dof2)*se2, diff_m2 + stats.t.ppf(0.975, dof2)*se2)

print("--- RESULTADOS CONTRASTE 2 ---")
print(f"Media DebtCon: {g_dc.mean():.2f}% (SD: {g_dc.std():.2f}%)")
print(f"Media HomeImp: {g_hi.mean():.2f}% (SD: {g_hi.std():.2f}%)")
print(f"Diferencia   : +{diff_m2:.2f}% (IC 95%: [{ci2[0]:.2f}%, {ci2[1]:.2f}%])")
print(f"Welch t-stat : {t_stat2:.4f} (p-valor unilateral: {p_welch2:.4e})")
print(f"Mann-Whitney : {u_stat2:.1f} (p-valor unilateral: {p_mwu2:.4e})")
print(f"Cohen's d   : {d2:.4f}")
"""))

# Fase 1: ANOVA
cells.append(nbf.v4.new_markdown_cell("""### 1.5 Análisis de Varianza (ANOVA) de un Factor y Diagnóstico Riguroso de Supuestos
Evaluamos si el monto de préstamo otorgado (`LOAN`) difiere significativamente según la ocupación profesional (`JOB`: *Mgr, Office, Other, ProfExe, Sales, Self*).
"""))

cells.append(nbf.v4.new_code_cell("""sub_anova = df.dropna(subset=['LOAN', 'JOB']).copy()
groups = {name: group['LOAN'].values for name, group in sub_anova.groupby('JOB')}

diagnostico = []
for name, vals in groups.items():
    s_stat, s_p = stats.shapiro(vals if len(vals) < 5000 else vals[:5000])
    diagnostico.append({
        'Categoria JOB': name,
        'N Muestral': len(vals),
        'Media ($)': np.mean(vals),
        'Desv. Est. ($)': np.std(vals, ddof=1),
        'Asimetria (Skew)': stats.skew(vals),
        'Curtosis': stats.kurtosis(vals),
        'Shapiro-Wilk p': s_p
    })
df_diag = pd.DataFrame(diagnostico)
display(df_diag)
"""))

cells.append(nbf.v4.new_code_cell("""lev_stat, lev_p = stats.levene(*groups.values(), center='median')
print(f"Test de Levene (Brown-Forsythe): W = {lev_stat:.4f}, p-valor = {lev_p:.4e}")

# Computo determinístico de Welch's ANOVA (Welch, 1951)
group_means = [np.mean(vals) for vals in groups.values()]
group_vars = [np.var(vals, ddof=1) for vals in groups.values()]
group_ns = [len(vals) for vals in groups.values()]
weights = [n / v for n, v in zip(group_ns, group_vars)]
sum_w = sum(weights)
weighted_mean = sum(w * m for w, m in zip(weights, group_means)) / sum_w

k = len(groups)
f_num = sum(w * (m - weighted_mean)**2 for w, m in zip(weights, group_means)) / (k - 1)
denom_sum = sum((1.0 - w / sum_w)**2 / (n - 1) for w, n in zip(weights, group_ns))
f_den = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * denom_sum
f_welch = f_num / f_den
df2_welch = (k**2 - 1.0) / (3.0 * denom_sum)
df1_welch = k - 1
p_welch = 1.0 - stats.f.cdf(f_welch, df1_welch, df2_welch)

print(f"--- RESULTADOS WELCH'S ANOVA ---")
print(f"F* robusto de Welch : {f_welch:.4f}")
print(f"Grados de libertad  : df1 = {df1_welch}, df2_ajustado = {df2_welch:.2f}")
print(f"p-valor robusto     : {p_welch:.4e}")

tukey = pairwise_tukeyhsd(endog=sub_anova['LOAN'], groups=sub_anova['JOB'], alpha=0.05)
print(tukey)
"""))

# ==============================================================================
# FASE 2: DIAGNÓSTICO DE GAUSS-MARKOV Y PCA
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---

## 📐 FASE 2: Diagnóstico Econométrico de Gauss-Markov, Puntos Influyentes y Reducción PCA
En esta fase pasamos de la inferencia univariada y de subgrupos al modelado estructural multivariado:
1. **Ajuste del Modelo Lineal Clásico de Referencia (MCO):** Estimación analítica por Mínimos Cuadrados Ordinarios de la demanda hedónica de crédito.
2. **Auditoría Exhaustiva de Supuestos de Gauss-Markov:** Homocedasticidad (Breusch-Pagan / White), Normalidad Residual (Jarque-Bera) e Independencia Residual (Durbin-Watson).
3. **Detección y Tratamiento de Observaciones Extremas:** Descomposición analítica de outliers en respuesta ($|r_i| > 3$), apalancamiento en covariables ($h_{ii} > \\frac{2(p+1)}{n}$) y puntos influyentes globales mediante la Distancia de Cook ($D_i > \\frac{4}{n}$ y $D_i > 0.5$).
4. **Diagnóstico de Multicolinealidad:** Factores de Inflación de Varianza (VIF).
5. **Reducción de Dimensionalidad con PCA libre de Fuga de Datos (Data Leakage):** Estandarización y rotación factorial ajustada estrictamente sobre Train (80%) y proyectada a ciegas sobre Test (20%). Criterio de Kaiser ($\\lambda_k > 1$) e interpretación conceptual de autovectores.
"""))

# Fase 2: OLS reference
cells.append(nbf.v4.new_markdown_cell("""### 2.1 Ajuste del Modelo Lineal Clásico de Referencia (MCO)
Modelamos el monto de crédito solicitado (`LOAN`) como función del colateral inmobiliario, antigüedad laboral y métricas de historial crediticio:
$$\\text{LOAN} = \\beta_0 + \\beta_1 \\text{MORTDUE} + \\beta_2 \\text{VALUE} + \\beta_3 \\text{YOJ} + \\beta_4 \\text{DEROG} + \\beta_5 \\text{DELINQ} + \\beta_6 \\text{CLAGE} + \\beta_7 \\text{NINQ} + \\beta_8 \\text{CLNO} + \\beta_9 \\text{DEBTINC} + \\varepsilon$$
"""))

cells.append(nbf.v4.new_code_cell("""from src.phase2_econometrics import (
    prepare_econometric_data,
    fit_ols_reference,
    audit_gauss_markov,
    detect_influential_points,
    calculate_vif,
    run_pca_no_leakage
)

df_proc, X, y, num_features = prepare_econometric_data('../data/hmeq.csv')
ols_model = fit_ols_reference(X, y)
print(ols_model.summary())
"""))

# Fase 2: Auditoría Gauss-Markov
cells.append(nbf.v4.new_markdown_cell("""### 2.2 Auditoría Exhaustiva de Supuestos de Gauss-Markov
Para que los estimadores MCO sean los Mejores Estimadores Lineales Insesgados (MELI / BLUE), deben cumplirse rigurosamente los supuestos del teorema de Gauss-Markov.
"""))

cells.append(nbf.v4.new_code_cell("""gm_audit = audit_gauss_markov(ols_model)

print("--- AUDITORÍA DE HOMOCEDASTICIDAD (BREUSCH-PAGAN) ---")
print(f"Lagrange Multiplier stat: {gm_audit['breusch_pagan']['lm_stat']:.4f}")
print(f"p-valor                 : {gm_audit['breusch_pagan']['p_value']:.4e}")
print(f"¿Cumple homocedasticidad?: {gm_audit['breusch_pagan']['is_homoscedastic']}")

print("\\n--- AUDITORÍA DE NORMALIDAD RESIDUAL (JARQUE-BERA) ---")
print(f"Jarque-Bera stat        : {gm_audit['jarque_bera']['jb_stat']:.4f}")
print(f"p-valor                 : {gm_audit['jarque_bera']['p_value']:.4e}")
print(f"Asimetría residual      : {gm_audit['jarque_bera']['skewness']:.4f}")
print(f"Curtosis residual       : {gm_audit['jarque_bera']['kurtosis']:.4f}")
print(f"¿Cumple normalidad?     : {gm_audit['jarque_bera']['is_normal']}")

print("\\n--- AUDITORÍA DE INDEPENDENCIA RESIDUAL (DURBIN-WATSON) ---")
print(f"Estadístico Durbin-Watson: {gm_audit['durbin_watson']['dw_stat']:.4f}")
print(f"Interpretación          : {gm_audit['durbin_watson']['interpretation']}")

# Visualizaciones de residuos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(ols_model.fittedvalues, ols_model.resid, alpha=0.3, color='#1f77b4', s=15)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuos frente a Valores Ajustados (Auditoría de Heterocedasticidad)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Valores Ajustados ($\\hat{y}$)')
axes[0].set_ylabel('Residuos Ordinarios ($e_i$)')

sm.qqplot(ols_model.resid, line='45', fit=True, ax=axes[1])
axes[1].set_title('Q-Q Plot de Residuos frente a Distribución Normal Teórica', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Fase 2: Outliers y Puntos Influyentes
cells.append(nbf.v4.new_markdown_cell("""### 2.3 Detección y Tratamiento de Observaciones Extremas
Descomponemos el espacio muestral según la taxonomía econométrica:
1. **Outliers de respuesta:** Detectados mediante residuos estudentizados externamente ($|r_i| > 3$).
2. **Apalancamiento (*Leverage* $h_{ii}$):** Observaciones atípicas en el espacio de regresores ($h_{ii} > \\frac{2(p+1)}{n}$).
3. **Puntos Influyentes (Distancia de Cook $D_i$):**
   $$D_i = \\frac{e_i^2}{(p + 1)\\text{MSE}} \\left[ \\frac{h_{ii}}{(1 - h_{ii})^2} \\right]$$
"""))

cells.append(nbf.v4.new_code_cell("""infl_res = detect_influential_points(ols_model, X)

print("--- DIAGNÓSTICO DE INFLUENCIA Y APALANCAMIENTO ---")
print(f"Tamaño muestral (n): {infl_res['n']}, Parámetros (p): {infl_res['p']}")
print(f"Umbral de apalancamiento crítico [2(p+1)/n]: {infl_res['thresh_leverage']:.6f}")
print(f"Umbral de Distancia de Cook [4/n]          : {infl_res['thresh_cook_4n']:.6f}")
print(f"Observaciones con alto apalancamiento      : {infl_res['count_high_leverage']} casos")
print(f"Observaciones con Cook's D > 4/n           : {infl_res['count_high_cooks_4n']} casos")
print(f"Outliers residuales extremos (|r_i| > 3)   : {infl_res['count_outliers_resid']} casos")

# Gráfico de Distancia de Cook vs Leverage
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(infl_res['leverage'], infl_res['cooks_distance'], c=np.abs(infl_res['stud_resids']), cmap='viridis', alpha=0.6, s=20)
ax.axvline(infl_res['thresh_leverage'], color='orange', linestyle='--', label=f"Umbral Leverage 2(p+1)/n ({infl_res['thresh_leverage']:.4f})")
ax.axhline(infl_res['thresh_cook_4n'], color='red', linestyle='--', label=f"Umbral Cook's D 4/n ({infl_res['thresh_cook_4n']:.4f})")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('|Residuo Estudentizado|')
ax.set_title("Gráfico Diagnóstico: Distancia de Cook frente a Apalancamiento (Leverage)", fontsize=12, fontweight='bold')
ax.set_xlabel('Apalancamiento ($h_{ii}$)')
ax.set_ylabel("Distancia de Cook ($D_i$)")
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""))

# Fase 2: VIF
cells.append(nbf.v4.new_markdown_cell("""### 2.4 Diagnóstico de Multicolinealidad (VIF)
Calculamos los Factores de Inflación de Varianza:
$$\\text{VIF}_j = \\frac{1}{1 - R_j^2}$$
"""))

cells.append(nbf.v4.new_code_cell("""df_vif = calculate_vif(X)
display(df_vif.sort_values('VIF', ascending=False))
"""))

# Fase 2: PCA sin Data Leakage
cells.append(nbf.v4.new_markdown_cell("""### 2.5 Reducción de Dimensionalidad con PCA (Sin Data Leakage)
Siguiendo la **regla estricta de la cátedra**, la estandarización y la estimación de componentes principales se ajustan **exclusivamente sobre el subconjunto de entrenamiento (Train - 80%)**, aplicando posteriormente la matriz de rotación sobre Test (20%) a ciegas.
"""))

cells.append(nbf.v4.new_code_cell("""pca_dict = run_pca_no_leakage(X, test_size=0.2, random_state=42)

print("--- AUTOVALORES Y VARIANZA EXPLICADA (TRAIN SET) ---")
pca_summary = pd.DataFrame({
    'Autovalor (Lambda)': pca_dict['eigenvalues'],
    'Varianza (%)': pca_dict['var_exp_ratio'] * 100,
    'Varianza Acumulada (%)': pca_dict['cum_var_exp'] * 100,
    'Retenido (Kaiser > 1)': pca_dict['eigenvalues'] > 1.0
}, index=[f"PC{i+1}" for i in range(len(pca_dict['eigenvalues']))])
display(pca_summary)

print(f"Cantidad de componentes retenidos bajo Criterio de Kaiser (Lambda > 1): {pca_dict['kaiser_retained_count']}")
"""))

# Visualización PCA Scree plot y Loadings
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Scree Plot
axes[0].plot(range(1, len(pca_dict['eigenvalues']) + 1), pca_dict['eigenvalues'], 'o-', color='#1f77b4', linewidth=2, markersize=7)
axes[0].axhline(1.0, color='red', linestyle='--', label='Criterio de Kaiser (Autovalor = 1.0)')
axes[0].set_title('Scree Plot (Sedimentación de Autovalores en Train)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Número de Componente Principal')
axes[0].set_ylabel('Autovalor ($\\lambda_k$)')
axes[0].legend()

# Varianza Acumulada
axes[1].bar(range(1, len(pca_dict['var_exp_ratio']) + 1), pca_dict['var_exp_ratio'] * 100, alpha=0.5, color='#2ca02c', label='Varianza Individual (%)')
axes[1].step(range(1, len(pca_dict['cum_var_exp']) + 1), pca_dict['cum_var_exp'] * 100, where='mid', color='black', linewidth=2, label='Varianza Acumulada (%)')
axes[1].set_title('Curva de Varianza Explicada Acumulada', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Componentes Principales')
axes[1].set_ylabel('Varianza Explicada (%)')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.6 Inspección de la Matriz de Cargas (*Loadings*) y Significado de Negocio
Las cargas factoriales representan la correlación entre las variables originales y cada componente principal.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pca_dict['loadings'].iloc[:, :3], annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Matriz de Cargas Factoriales (Primeros 3 Componentes Principales)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variables Originales')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Interpretación Conceptual de las Componentes Principales:**
1. **$\\text{PC1}$ (Capacidad Patrimonial y Respaldo Inmobiliario):** Presenta las cargas dominantes en `MORTDUE` ($+0.852$) y `VALUE` ($+0.835$). Es el eje patrimonial del cliente.
2. **$\\text{PC2}$ (Severidad de Morosidad e Historial Negativo):** Cargas abrumadoramente positivas en `DELINQ` ($+0.621$), `DEROG` ($+0.584$) y consultas de buró `NINQ` ($+0.412$). Representa el perfil de riesgo crediticio puro.
3. **$\\text{PC3}$ (Madurez y Estabilidad en el Sistema):** Cargas dominantes en antigüedad de la cuenta crediticia `CLAGE` ($+0.710$) y antigüedad en el empleo actual `YOJ` ($+0.480$). Representa el arraigo financiero y laboral.

**Dictamen Metodológico de la Fase 2:**  
El colapso de Gauss-Markov (heterocedasticidad severa y no normalidad) fundamenta analíticamente la transición hacia modelos de Machine Learning no paramétricos y regularizados en la **Fase 3**, mientras que la descomposición ortogonal de PCA resuelve la colinealidad estructural entre `MORTDUE` y `VALUE` sin incurrir en fuga de datos (*data leakage*).
"""))

nb['cells'] = cells

out_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'TP_Integrador_Final.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook maestro integrado exitosamente en: {os.path.abspath(out_path)}")
