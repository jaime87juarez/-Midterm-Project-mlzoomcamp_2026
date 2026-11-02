# ============================================================
# SERVICIO DE PREDICCIÓN - RANDOM FOREST
# ============================================================

from pathlib import Path

import joblib
import pandas as pd


# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_model.joblib"


# ------------------------------------------------------------
# 2. CARGAR MODELO ENTRENADO
# ------------------------------------------------------------

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]

features = artifact["features"]


# ------------------------------------------------------------
# 3. IDENTIFICAR VARIABLES NUMÉRICAS
# ------------------------------------------------------------
# Estas columnas representan valores numéricos.
#
# Las convertiremos a float64 para evitar incompatibilidades
# entre int64 y float64 durante el preprocesamiento.

NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "arrival_month_num",
    "total_nights",
    "total_guests"
]


# ------------------------------------------------------------
# 4. FUNCIÓN DE PREDICCIÓN
# ------------------------------------------------------------

def predict(data: dict):

    """
    Recibe las características de una reserva hotelera.

    Convierte las variables numéricas a float64.

    Utiliza el modelo Random Forest entrenado.

    Devuelve el ADR estimado.
    """

    # Convertir entrada JSON a DataFrame
    df = pd.DataFrame([data])

    # --------------------------------------------------------
    # VALIDAR COLUMNAS
    # --------------------------------------------------------

    missing = [
        col for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan variables requeridas: {missing}"
        )

    # --------------------------------------------------------
    # CONVERTIR VARIABLES NUMÉRICAS
    # --------------------------------------------------------

    for col in NUMERIC_FEATURES:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="raise"
            ).astype("float64")

    # --------------------------------------------------------
    # ORDENAR VARIABLES
    # --------------------------------------------------------

    df = df[features]

    # --------------------------------------------------------
    # GENERAR PREDICCIÓN
    # --------------------------------------------------------

    prediction = model.predict(df)

    return float(prediction[0])
