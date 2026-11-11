"""
tests/test_data_pipeline.py
Auditoría del Contrato de Datos, Limpieza y Partición sin Data Leakage
Auditor: QA & Validation Engineer
"""

import pytest
import pandas as pd
import numpy as np
import json
from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    METADATA_PATH,
    TARGET_BINARY,
    TREATMENT_VAR,
    ANOVA_FACTOR,
    NUMERICAL_FEATURES
)


def test_processed_files_exist():
    """Verifica que los archivos oficiales procesados existan en disco."""
    assert TRAIN_PATH.exists(), f"Falta {TRAIN_PATH}"
    assert TEST_PATH.exists(), f"Falta {TEST_PATH}"
    assert METADATA_PATH.exists(), f"Falta {METADATA_PATH}"


def test_no_missing_values_and_shapes():
    """Verifica ausencia total de nulos y dimensiones de la partición 80/20."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    assert train_df.isnull().sum().sum() == 0, "Hay valores nulos en Train."
    assert test_df.isnull().sum().sum() == 0, "Hay valores nulos en Test."
    
    assert len(train_df) == 4768
    assert len(test_df) == 1192
    assert len(train_df) + len(test_df) == 5960


def test_stratified_target_distribution():
    """Verifica que la estratificación de BAD sea casi idéntica en Train y Test."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_rate = train_df[TARGET_BINARY].mean()
    test_rate = test_df[TARGET_BINARY].mean()
    
    # La diferencia en proporción no debe superar el 0.5%
    assert abs(train_rate - test_rate) < 0.005
    assert set(train_df[TARGET_BINARY].unique()) == {0, 1}


def test_treatment_variable_definition():
    """Verifica que el tratamiento T esté binarizado (0 o 1) y presente en ambos splits."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    assert TREATMENT_VAR in train_df.columns
    assert TREATMENT_VAR in test_df.columns
    assert set(train_df[TREATMENT_VAR].unique()).issubset({0, 1})
    assert set(test_df[TREATMENT_VAR].unique()).issubset({0, 1})


def test_anova_factor_categories():
    """Verifica que el factor de ANOVA (JOB) tenga al menos 3 categorías en ambos splits."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    categories = train_df[ANOVA_FACTOR].unique()
    assert len(categories) >= 3, "Se requieren al menos 3 categorías para el ANOVA."
    assert set(categories) == set(test_df[ANOVA_FACTOR].unique())
