"""
src/data_pipeline.py
Pipeline Canónico de Limpieza, Imputación sin Fuga y Partición Estratificada
Genera los datasets oficiales de trabajo: data/processed/train.csv y data/processed/test.csv
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW_PATH,
    DATA_PROCESSED_DIR,
    TRAIN_PATH,
    TEST_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_BINARY,
    TARGET_CONTINUOUS,
    TREATMENT_VAR,
    ORIGINAL_TREATMENT_COL,
    ANOVA_FACTOR,
    ANOVA_VARIABLE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Ejecuta el pipeline determinístico de preparación de datos:
    1. Carga del dataset crudo.
    2. Construcción formal del Tratamiento (T) e indicadores de omisión MAR/MNAR.
    3. Partición estratificada Train (80%) y Test (20%).
    4. Imputación de nulos aprendida estrictamente en Train y aplicada a Test (cero data leakage).
    5. Serialización a disco para todo el equipo.
    """
    print(f"[PIPELINE] Leyendo dataset crudo desde: {DATA_RAW_PATH}")
    df_raw = pd.read_csv(DATA_RAW_PATH)
    n_raw, p_raw = df_raw.shape
    print(f"[PIPELINE] Dimensiones iniciales: {n_raw} filas x {p_raw} columnas")

    df = df_raw.copy()

    # 1. Definición formal del Tratamiento Causal T (REASON binarizado)
    # DebtCon (Consolidación de Deuda) = 1, HomeImp (Mejoras Hogar) = 0
    # Imputamos nulos de REASON con la moda ('DebtCon') antes de binarizar
    mode_reason = df[ORIGINAL_TREATMENT_COL].mode()[0]
    df[ORIGINAL_TREATMENT_COL] = df[ORIGINAL_TREATMENT_COL].fillna(mode_reason)
    df[TREATMENT_VAR] = (df[ORIGINAL_TREATMENT_COL] == "DebtCon").astype(int)

    # 2. Imputación de categoría ocupacional JOB con 'Other' (categoría mayoritaria)
    df[ANOVA_FACTOR] = df[ANOVA_FACTOR].fillna("Other")

    # 3. Indicador explícito de pérdida para DEBTINC (fundamentado por MAR/MNAR en Fase 1)
    df["DEBTINC_is_missing"] = df["DEBTINC"].isnull().astype(int)

    # 4. Partición estratificada Train / Test (80% / 20%) basada en el Target Y (BAD)
    print(f"[PIPELINE] Particionando datos (Train {1-TEST_SIZE:.0%} / Test {TEST_SIZE:.0%}) con seed={RANDOM_STATE}...")
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_BINARY]
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # 5. Imputación sin fuga de datos (Data Leakage):
    # Se calculan los estadísticos de imputación (medianas) ÚNICAMENTE sobre train_df
    num_cols_to_impute = [col for col in NUMERICAL_FEATURES if col != TARGET_CONTINUOUS]
    # También incluimos LOAN si tuviera nulos (HMEQ tiene 0 nulos en LOAN)
    medians_learned = {}
    for col in NUMERICAL_FEATURES:
        median_val = float(train_df[col].median())
        medians_learned[col] = median_val
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    # 6. Verificación de integridad: cero nulos remanentes
    assert train_df.isnull().sum().sum() == 0, "Nulos remanentes detectados en train_df."
    assert test_df.isnull().sum().sum() == 0, "Nulos remanentes detectados en test_df."

    # 7. Serialización a disco
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"[PIPELINE] Dataset de entrenamiento guardado en: {TRAIN_PATH} ({len(train_df)} filas)")
    print(f"[PIPELINE] Dataset de prueba guardado en: {TEST_PATH} ({len(test_df)} filas)")

    # Metadata de gobernanza de datos para el equipo
    metadata = {
        "dataset_name": "HMEQ (Home Equity Loan Dataset)",
        "random_state": RANDOM_STATE,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "target_binary": {
            "name": TARGET_BINARY,
            "train_rate": float(train_df[TARGET_BINARY].mean()),
            "test_rate": float(test_df[TARGET_BINARY].mean())
        },
        "target_continuous": {
            "name": TARGET_CONTINUOUS,
            "train_mean": float(train_df[TARGET_CONTINUOUS].mean()),
            "test_mean": float(test_df[TARGET_CONTINUOUS].mean())
        },
        "treatment": {
            "name": TREATMENT_VAR,
            "description": "1 = DebtCon (Consolidación de Deuda), 0 = HomeImp (Mejoras Hogar)",
            "train_rate": float(train_df[TREATMENT_VAR].mean()),
            "test_rate": float(test_df[TREATMENT_VAR].mean())
        },
        "anova_design": {
            "factor": ANOVA_FACTOR,
            "continuous_var": ANOVA_VARIABLE,
            "categories": sorted(train_df[ANOVA_FACTOR].unique().tolist())
        },
        "medians_learned_train": medians_learned
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"[PIPELINE] Metadatos y contrato guardados en: {METADATA_PATH}")

    return train_df, test_df, metadata


if __name__ == "__main__":
    train_df, test_df, meta = run_pipeline()
    print("\n" + "="*70)
    print("CONTRATO DE DATOS LISTO PARA DISTRIBUCIÓN DEL EQUIPO")
    print("="*70)
    print(f"Train Shape: {train_df.shape} (Default rate: {meta['target_binary']['train_rate']:.2%})")
    print(f"Test Shape : {test_df.shape}  (Default rate: {meta['target_binary']['test_rate']:.2%})")
    print(f"Tratamiento T (DebtCon rate): {meta['treatment']['train_rate']:.2%}")
    print(f"Grupos ANOVA ({meta['anova_design']['factor']}): {meta['anova_design']['categories']}")
    print("="*70)
