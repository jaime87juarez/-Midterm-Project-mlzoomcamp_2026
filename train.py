# ============================================================
# ENTRENAMIENTO DEL MODELO FINAL
# ============================================================

from pathlib import Path
import sys

import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from pipeline import (
    load_data,
    prepare,
    split_data,
    feature_columns,
    make_preprocessor
)


# ------------------------------------------------------------
# 2. CONFIGURACIÓN DEL MODELO
# ------------------------------------------------------------

MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "final_model.joblib"


# ------------------------------------------------------------
# 3. FUNCIÓN DE ENTRENAMIENTO
# ------------------------------------------------------------

def train():

    print("Iniciando entrenamiento...")

    # Cargar información
    raw = load_data()

    # Preparar información
    df = prepare(raw)

    # Filtrar reservas válidas
    df = df[
        (df["is_canceled"] == 0) &
        (df["adr"].between(0, 1000)) &
        (df["total_nights"] > 0)
    ].copy()

    # Dividir datos
    train_df, val_df, test_df = split_data(df)

    # Identificar variables predictoras
    cols = feature_columns(
        df,
        "regression"
    )

    # Crear pipeline del modelo final
    model = Pipeline([

        (
            "prep",
            make_preprocessor(train_df[cols])
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=5,
                random_state=1,
                n_jobs=-1
            )
        )

    ])

    # Entrenar el modelo
    model.fit(
        train_df[cols],
        train_df["adr"]
    )

    # Guardar modelo y columnas requeridas
    artifact = {
        "model": model,
        "features": cols,
        "model_name": "Random Forest",
        "target": "adr"
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    print("Entrenamiento completado.")
    print(f"Modelo guardado en: {MODEL_PATH}")


# ------------------------------------------------------------
# 4. EJECUCIÓN
# ------------------------------------------------------------

if __name__ == "__main__":
    train()
