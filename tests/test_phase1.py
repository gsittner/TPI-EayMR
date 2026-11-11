"""
tests/test_phase1.py
Tests automáticos de calidad determinística para la Fase 1
Auditor: QA & Validation Engineer
"""

import pytest
import os
import pandas as pd
import numpy as np
from src.phase1_inference import (
    load_raw_data,
    analyze_missing_mechanism,
    run_hypothesis_contrasts,
    run_anova_and_diagnostics
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hmeq.csv")


def test_dataset_loading_and_shape():
    """Verifica que el dataset exista y cumpla N >= 1500 y >= 10 variables."""
    assert os.path.exists(DATA_PATH), f"El archivo {DATA_PATH} no existe."
    df = load_raw_data(DATA_PATH)
    assert len(df) >= 1500, f"N insuficiente: {len(df)} < 1500"
    assert df.shape[1] >= 10, f"Variables insuficientes: {df.shape[1]} < 10"


def test_target_distribution():
    """Verifica que la variable objetivo BAD sea binaria."""
    df = load_raw_data(DATA_PATH)
    assert "BAD" in df.columns
    unique_vals = set(df["BAD"].dropna().unique())
    assert unique_vals.issubset({0, 1})


def test_missing_data_classification():
    """Verifica que el modelo logit detecte la no-aleatoriedad de DEBTINC (rechazo MCAR)."""
    df = load_raw_data(DATA_PATH)
    res = analyze_missing_mechanism(df)
    logit = res["logit_debtinc_missing"]
    # El p-valor para BAD debe ser altamente significativo (< 0.001)
    p_val_bad = logit.pvalues["BAD"]
    assert p_val_bad < 0.001, "El patrón de omisión no muestra dependencia significativa con BAD."


def test_hypothesis_contrasts_structure():
    """Verifica que los contrastes retornen p-valores, estadísticos y tamaños de efecto."""
    df = load_raw_data(DATA_PATH)
    contrasts = run_hypothesis_contrasts(df)
    
    assert "contraste_1" in contrasts
    c1 = contrasts["contraste_1"]
    assert c1["p_value_welch"] < 0.05
    assert not np.isnan(c1["cohen_d"])
    
    assert "contraste_2" in contrasts
    c2 = contrasts["contraste_2"]
    assert c2["p_value_welch"] < 0.05
    assert not np.isnan(c2["cohen_d"])


def test_anova_heteroscedasticity_and_welch():
    """Verifica la detección de heterocedasticidad (Levene) y cálculo de Welch ANOVA."""
    df = load_raw_data(DATA_PATH)
    anova_res = run_anova_and_diagnostics(df)
    
    # El test de Levene debe rechazar homocedasticidad
    assert anova_res["levene_test"]["p_value"] < 0.01
    
    # Welch ANOVA debe ejecutarse con éxito
    welch = anova_res["welch_anova"]
    assert welch["p_value"] < 0.001
    assert welch["df1"] == 5
