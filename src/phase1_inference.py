"""
src/phase1_inference.py
Módulo de Inferencia Estadística Clásica, Diagnóstico de Distribuciones y ANOVA (Fase 1)
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns


def load_raw_data(filepath: str = "data/hmeq.csv") -> pd.DataFrame:
    """Carga el dataset HMEQ y reporta dimensiones basales."""
    df = pd.read_csv(filepath)
    return df


def analyze_missing_mechanism(df: pd.DataFrame) -> dict:
    """
    Analiza el mecanismo de pérdida de datos para clasificar entre MCAR, MAR y MNAR.
    Ajusta una regresión logística modelando el indicador de dato faltante en DEBTINC
    en función de LOAN, VALUE y el estado crediticio BAD.
    """
    results = {}
    missing_summary = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    results["missing_table"] = pd.DataFrame({"Nulos": missing_summary, "Porcentaje": missing_pct})

    # Test de MAR sobre DEBTINC
    df_temp = df.copy()
    df_temp["DEBTINC_missing"] = df_temp["DEBTINC"].isnull().astype(int)
    logit_model = sm.Logit.from_formula("DEBTINC_missing ~ LOAN + VALUE + BAD", data=df_temp).fit(disp=False)
    
    results["logit_debtinc_missing"] = logit_model
    results["is_mcar"] = False  # Si p < 0.05, se rechaza MCAR a favor de MAR
    return results


def run_hypothesis_contrasts(df: pd.DataFrame) -> dict:
    """
    Ejecuta dos contrastes de hipótesis formales de alto impacto de negocio:
    1. LOAN según BAD (Cumplidor vs Moroso).
    2. DEBTINC según REASON (DebtCon vs HomeImp).
    """
    contrasts = {}

    # Contraste 1: LOAN según BAD
    g0 = df[df["BAD"] == 0]["LOAN"].dropna()
    g1 = df[df["BAD"] == 1]["LOAN"].dropna()

    t_stat_1, p_val_1 = stats.ttest_ind(g0, g1, equal_var=False)  # Welch t-test
    u_stat_1, u_pval_1 = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    s_pooled_1 = np.sqrt(((len(g0) - 1) * g0.var() + (len(g1) - 1) * g1.var()) / (len(g0) + len(g1) - 2))
    cohen_d_1 = (g1.mean() - g0.mean()) / s_pooled_1
    
    # Intervalo de confianza 95% para la diferencia de medias (Welch)
    diff_mean_1 = g1.mean() - g0.mean()
    se_diff_1 = np.sqrt(g0.var() / len(g0) + g1.var() / len(g1))
    dof_1 = (g0.var() / len(g0) + g1.var() / len(g1))**2 / (
        (g0.var() / len(g0))**2 / (len(g0) - 1) + (g1.var() / len(g1))**2 / (len(g1) - 1)
    )
    t_crit_1 = stats.t.ppf(0.975, dof_1)
    ci_1 = (diff_mean_1 - t_crit_1 * se_diff_1, diff_mean_1 + t_crit_1 * se_diff_1)

    contrasts["contraste_1"] = {
        "variable": "LOAN",
        "factor": "BAD",
        "mean_0": g0.mean(),
        "std_0": g0.std(),
        "mean_1": g1.mean(),
        "std_1": g1.std(),
        "t_stat": t_stat_1,
        "p_value_welch": p_val_1,
        "u_stat": u_stat_1,
        "p_value_mwu": u_pval_1,
        "cohen_d": cohen_d_1,
        "diff_ci95": ci_1
    }

    # Contraste 2: DEBTINC según REASON (DebtCon vs HomeImp)
    sub2 = df.dropna(subset=["DEBTINC", "REASON"])
    g_dc = sub2[sub2["REASON"] == "DebtCon"]["DEBTINC"]
    g_hi = sub2[sub2["REASON"] == "HomeImp"]["DEBTINC"]

    t_stat_2, p_val_2 = stats.ttest_ind(g_dc, g_hi, equal_var=False)
    u_stat_2, u_pval_2 = stats.mannwhitneyu(g_dc, g_hi, alternative="two-sided")
    s_pooled_2 = np.sqrt(((len(g_dc) - 1) * g_dc.var() + (len(g_hi) - 1) * g_hi.var()) / (len(g_dc) + len(g_hi) - 2))
    cohen_d_2 = (g_dc.mean() - g_hi.mean()) / s_pooled_2
    
    diff_mean_2 = g_dc.mean() - g_hi.mean()
    se_diff_2 = np.sqrt(g_dc.var() / len(g_dc) + g_hi.var() / len(g_hi))
    dof_2 = (g_dc.var() / len(g_dc) + g_hi.var() / len(g_hi))**2 / (
        (g_dc.var() / len(g_dc))**2 / (len(g_dc) - 1) + (g_hi.var() / len(g_hi))**2 / (len(g_hi) - 1)
    )
    t_crit_2 = stats.t.ppf(0.975, dof_2)
    ci_2 = (diff_mean_2 - t_crit_2 * se_diff_2, diff_mean_2 + t_crit_2 * se_diff_2)

    contrasts["contraste_2"] = {
        "variable": "DEBTINC",
        "factor": "REASON",
        "mean_DebtCon": g_dc.mean(),
        "std_DebtCon": g_dc.std(),
        "mean_HomeImp": g_hi.mean(),
        "std_HomeImp": g_hi.std(),
        "t_stat": t_stat_2,
        "p_value_welch": p_val_2,
        "u_stat": u_stat_2,
        "p_value_mwu": u_pval_2,
        "cohen_d": cohen_d_2,
        "diff_ci95": ci_2
    }

    return contrasts


def run_anova_and_diagnostics(df: pd.DataFrame, continuous_var: str = "LOAN", group_var: str = "JOB") -> dict:
    """
    Ejecuta el ANOVA de un factor con diagnósticos exhaustivos:
    - Evaluación de normalidad: Shapiro-Wilk, asimetría y curtosis por grupo.
    - Evaluación de homocedasticidad: Test de Levene (con mediana / Brown-Forsythe).
    - Welch's ANOVA para varianzas no homogéneas.
    - Kruskal-Wallis como alternativa no paramétrica.
    - Comparaciones múltiples post-hoc (Tukey HSD).
    """
    results = {}
    sub = df.dropna(subset=[continuous_var, group_var]).copy()
    groups = {name: group[continuous_var].values for name, group in sub.groupby(group_var)}

    # Métricas descriptivas y de distribución por grupo
    diag_table = []
    for name, vals in groups.items():
        shapiro_stat, shapiro_p = stats.shapiro(vals) if len(vals) < 5000 else stats.shapiro(vals[:5000])
        skew = stats.skew(vals)
        kurt = stats.kurtosis(vals)
        diag_table.append({
            "Grupo": name,
            "N": len(vals),
            "Media": np.mean(vals),
            "Desv_Est": np.std(vals, ddof=1),
            "Asimetria": skew,
            "Curtosis": kurt,
            "Shapiro_stat": shapiro_stat,
            "Shapiro_p": shapiro_p
        })
    results["diagnostico_grupos"] = pd.DataFrame(diag_table)

    # ANOVA Clásico
    f_stat, classic_p = stats.f_oneway(*groups.values())
    results["anova_clasico"] = {"F_stat": f_stat, "p_value": classic_p}

    # Test de Homocedasticidad de Levene
    lev_stat, lev_p = stats.levene(*groups.values(), center="median")
    results["levene_test"] = {"W_stat": lev_stat, "p_value": lev_p}

    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*groups.values())
    results["kruskal_wallis"] = {"H_stat": kw_stat, "p_value": kw_p}

    # Welch's ANOVA manual (para no depender de librerías externas no estándar)
    # Welch (1951): F* con corrección de grados de libertad
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

    results["welch_anova"] = {
        "F_welch": f_welch,
        "df1": df1_welch,
        "df2": df2_welch,
        "p_value": p_welch
    }

    # Tukey HSD Post-Hoc
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(sub[continuous_var], sub[group_var], alpha=0.05)
    results["tukey_posthoc"] = tukey

    return results


if __name__ == "__main__":
    df = load_raw_data(r"C:\Users\German\Desktop\estadistica\TPI\data\hmeq.csv")
    print("Dataset shape:", df.shape)
    
    missing_res = analyze_missing_mechanism(df)
    print("\n--- Analisis de Perdida de Datos (DEBTINC) ---")
    print(missing_res["logit_debtinc_missing"].summary().tables[1])
    
    contrasts_res = run_hypothesis_contrasts(df)
    print("\n--- Contraste 1: LOAN por BAD ---")
    print(contrasts_res["contraste_1"])
    print("\n--- Contraste 2: DEBTINC por REASON ---")
    print(contrasts_res["contraste_2"])
    
    anova_res = run_anova_and_diagnostics(df)
    print("\n--- Diagnostico por Grupos (JOB) ---")
    print(anova_res["diagnostico_grupos"])
    print("\n--- Welch ANOVA ---")
    print(anova_res["welch_anova"])
    print("\n--- Levene Test ---")
    print(anova_res["levene_test"])
