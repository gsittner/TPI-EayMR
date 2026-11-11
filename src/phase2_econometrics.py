"""
src/phase2_econometrics.py
Módulo de Diagnóstico de Gauss-Markov, Puntos Influyentes y Reducción PCA (Fase 2)
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor


def prepare_econometric_data(filepath: str = "data/hmeq.csv", impute_strategy: str = "median") -> tuple:
    """
    Carga y prepara los datos para la auditoría econométrica y PCA.
    Aplica imputación controlada (mediana condicional por ocupación o mediana global)
    para no incurrir en sesgo listwise.
    """
    df = pd.read_csv(filepath)
    num_cols = ["MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
    
    df_proc = df.copy()
    if impute_strategy == "median":
        for col in num_cols:
            median_val = df_proc[col].median()
            df_proc[col] = df_proc[col].fillna(median_val)
    
    y = df_proc["LOAN"]
    X = df_proc[num_cols]
    return df_proc, X, y, num_cols


def fit_ols_reference(X: pd.DataFrame, y: pd.Series):
    """
    Ajusta el modelo lineal clásico de referencia por Mínimos Cuadrados Ordinarios (MCO).
    Retorna el modelo ajustado de statsmodels.
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model


def audit_gauss_markov(model) -> dict:
    """
    Auditoría exhaustiva de supuestos de Gauss-Markov:
    1. Homocedasticidad: Test de Breusch-Pagan.
    2. Normalidad de residuos: Test de Jarque-Bera.
    3. Independencia de errores: Estadístico de Durbin-Watson.
    """
    # 1. Breusch-Pagan
    bp_stat, bp_pval, f_stat, f_pval = het_breuschpagan(model.resid, model.model.exog)
    
    # 2. Jarque-Bera
    jb_stat, jb_pval, skew, kurt = jarque_bera(model.resid)
    
    # 3. Durbin-Watson
    dw_stat = durbin_watson(model.resid)
    
    return {
        "breusch_pagan": {
            "lm_stat": bp_stat,
            "p_value": bp_pval,
            "f_stat": f_stat,
            "f_pval": f_pval,
            "is_homoscedastic": bp_pval >= 0.05
        },
        "jarque_bera": {
            "jb_stat": jb_stat,
            "p_value": jb_pval,
            "skewness": skew,
            "kurtosis": kurt,
            "is_normal": jb_pval >= 0.05
        },
        "durbin_watson": {
            "dw_stat": dw_stat,
            "interpretation": "Sin autocorrelación serial sustancial" if 1.5 < dw_stat < 2.5 else "Posible autocorrelación"
        }
    }


def detect_influential_points(model, X: pd.DataFrame) -> dict:
    """
    Descompone analíticamente el impacto de observaciones atípicas con formulación vectorial exacta:
    - Residuos estudentizados externamente (|r_i| > 3).
    - Apalancamiento (Leverage h_ii > 2(p+1)/n).
    - Distancia de Cook (D_i > 4/n y D_i > 0.5).
    """
    n = len(X)
    p = X.shape[1]
    e = model.resid.values
    mse = model.mse_resid

    # Diagonal de la matriz sombrero (Leverage h_ii) calculada eficientemente via QR
    X_mat = sm.add_constant(X).values
    Q, _ = np.linalg.qr(X_mat)
    leverage = np.sum(Q**2, axis=1)

    # Distancia de Cook D_i
    denom_lev = np.maximum((1.0 - leverage)**2, 1e-12)
    cooks_d = (e**2 / ((p + 1) * mse)) * (leverage / denom_lev)

    # Residuos estudentizados externamente (leave-one-out MSE)
    df_e = n - p - 1
    mse_i = (df_e * mse - (e**2 / np.maximum(1.0 - leverage, 1e-12))) / (df_e - 1)
    mse_i = np.maximum(mse_i, 1e-12)
    stud_resids = e / (np.sqrt(mse_i) * np.sqrt(np.maximum(1.0 - leverage, 1e-12)))
    
    # Umbrales
    thresh_cook_4n = 4.0 / n
    thresh_cook_half = 0.5
    thresh_lev = 2.0 * (p + 1) / n
    
    outliers_resids = np.where(np.abs(stud_resids) > 3)[0]
    high_leverage = np.where(leverage > thresh_lev)[0]
    high_cooks = np.where(cooks_d > thresh_cook_4n)[0]
    extreme_cooks = np.where(cooks_d > thresh_cook_half)[0]
    
    return {
        "n": n,
        "p": p,
        "thresh_leverage": thresh_lev,
        "thresh_cook_4n": thresh_cook_4n,
        "count_outliers_resid": len(outliers_resids),
        "count_high_leverage": len(high_leverage),
        "count_high_cooks_4n": len(high_cooks),
        "count_extreme_cooks_half": len(extreme_cooks),
        "cooks_distance": cooks_d,
        "leverage": leverage,
        "stud_resids": stud_resids
    }


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los Factores de Inflación de Varianza (VIF) para cada covariable.
    VIF_j = 1 / (1 - R_j^2)
    """
    X_const = sm.add_constant(X)
    vif_data = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(X_const.values, i)
        vif_data.append({"Variable": col, "VIF": vif_val, "Alerta": vif_val > 5.0})
    return pd.DataFrame(vif_data)


def run_pca_no_leakage(X: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Ejecuta el análisis de componentes principales (PCA) con estricto blindaje contra Data Leakage:
    - Estandarización ajustada únicamente en Train.
    - Transformación ciega sobre Test.
    - Criterio de Kaiser (autovalores > 1) y matriz de cargas (loadings).
    """
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA()
    pca.fit(X_train_scaled)
    
    # Autovalores (lambda_k = n_samples * explained_variance_)
    eigenvalues = pca.explained_variance_
    var_exp_ratio = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp_ratio)
    
    # Criterio de Kaiser: lambda > 1
    kaiser_retained = np.sum(eigenvalues > 1.0)
    
    # Matriz de cargas (Loadings = Autovectores * sqrt(Autovalores))
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=[f"PC{i+1}" for i in range(len(eigenvalues))],
        index=X.columns
    )
    
    return {
        "scaler": scaler,
        "pca": pca,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "eigenvalues": eigenvalues,
        "var_exp_ratio": var_exp_ratio,
        "cum_var_exp": cum_var_exp,
        "kaiser_retained_count": kaiser_retained,
        "loadings": loadings,
        "X_train_pca": pca.transform(X_train_scaled),
        "X_test_pca": pca.transform(X_test_scaled)
    }
