"""
tests/test_phase2.py
Tests automáticos de calidad determinística para la Fase 2 (Gauss-Markov y PCA)
Auditor: QA & Validation Engineer
"""

import pytest
import os
import pandas as pd
import numpy as np
from src.phase2_econometrics import (
    prepare_econometric_data,
    fit_ols_reference,
    audit_gauss_markov,
    detect_influential_points,
    calculate_vif,
    run_pca_no_leakage
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hmeq.csv")


def test_ols_reference_fit():
    """Verifica el ajuste correcto del modelo MCO de referencia y reporte de métricas."""
    _, X, y, _ = prepare_econometric_data(DATA_PATH)
    model = fit_ols_reference(X, y)
    
    assert model.rsquared > 0.10, "El R^2 del modelo MCO es inferior al umbral mínimo esperado."
    assert model.f_pvalue < 1e-10, "El test F global no resulta estadísticamente significativo."
    assert len(model.params) == X.shape[1] + 1, "Discrepancia en la cantidad de coeficientes estimados."


def test_gauss_markov_diagnostics():
    """Verifica que la auditoría de Gauss-Markov detecte heterocedasticidad y no normalidad."""
    _, X, y, _ = prepare_econometric_data(DATA_PATH)
    model = fit_ols_reference(X, y)
    diagnostics = audit_gauss_markov(model)
    
    # Breusch-Pagan debe rechazar homocedasticidad
    assert diagnostics["breusch_pagan"]["p_value"] < 0.05
    assert not diagnostics["breusch_pagan"]["is_homoscedastic"]
    
    # Jarque-Bera debe rechazar normalidad de residuos
    assert diagnostics["jarque_bera"]["p_value"] < 0.05
    assert not diagnostics["jarque_bera"]["is_normal"]
    assert diagnostics["jarque_bera"]["skewness"] > 1.0


def test_influential_points_and_cooks_distance():
    """Verifica el cálculo de residuos estudentizados, leverage y distancia de Cook."""
    _, X, y, _ = prepare_econometric_data(DATA_PATH)
    model = fit_ols_reference(X, y)
    infl_dict = detect_influential_points(model, X)
    
    assert infl_dict["count_outliers_resid"] > 0, "No se detectaron outliers con |r_i| > 3."
    assert infl_dict["count_high_leverage"] > 0, "No se detectaron puntos de alto apalancamiento."
    assert infl_dict["count_high_cooks_4n"] > 0, "No se detectaron observaciones influyentes según 4/n."
    assert np.all(infl_dict["leverage"] >= 0) and np.all(infl_dict["leverage"] <= 1)


def test_vif_calculation():
    """Verifica el cálculo de Factores de Inflación de Varianza (VIF)."""
    _, X, _, _ = prepare_econometric_data(DATA_PATH)
    df_vif = calculate_vif(X)
    
    assert len(df_vif) == X.shape[1]
    assert "MORTDUE" in df_vif["Variable"].values
    assert "VALUE" in df_vif["Variable"].values
    # MORTDUE y VALUE tienen la mayor colinealidad
    vif_mortdue = df_vif[df_vif["Variable"] == "MORTDUE"]["VIF"].values[0]
    assert vif_mortdue > 2.5, "MORTDUE debería exhibir el mayor factor de inflación de varianza."
    assert df_vif.loc[df_vif["VIF"].idxmax(), "Variable"] in ["MORTDUE", "VALUE"]


def test_pca_no_data_leakage():
    """Verifica que PCA aplique la regla estricta: fit solo en Train y transformación en Test."""
    _, X, _, _ = prepare_econometric_data(DATA_PATH)
    pca_results = run_pca_no_leakage(X, test_size=0.2, random_state=42)
    
    # Verificar dimensiones
    n_total = len(X)
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train
    
    assert pca_results["X_train_pca"].shape[0] == n_train
    assert pca_results["X_test_pca"].shape[0] == n_test
    assert pca_results["X_train_pca"].shape[1] == X.shape[1]
    
    # Criterio de Kaiser: autovalores ordenados descendentemente
    eigenvalues = pca_results["eigenvalues"]
    assert np.all(np.diff(eigenvalues) <= 0), "Los autovalores no están ordenados decrecientemente."
    assert pca_results["kaiser_retained_count"] >= 2, "Kaiser debe retener al menos 2 componentes."
