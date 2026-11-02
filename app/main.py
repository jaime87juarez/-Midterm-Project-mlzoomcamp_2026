# ============================================================
# API DE PREDICCIONES CON FASTAPI
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.predic import predict, features


# ------------------------------------------------------------
# 1. CREAR APLICACIÓN
# ------------------------------------------------------------

app = FastAPI(
    title="Hotel Yield ML API",
    description="API para predicción de ADR hotelero",
    version="1.0.0"
)


# ------------------------------------------------------------
# 2. MODELO DE ENTRADA
# ------------------------------------------------------------

class PredictionRequest(BaseModel):

    data: dict[str, str | int | float | bool | None]


# ------------------------------------------------------------
# 3. ENDPOINT DE BIENVENIDA
# ------------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Hotel Yield ML API",
        "status": "running"
    }


# ------------------------------------------------------------
# 4. ENDPOINT DE SALUD
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True
    }


# ------------------------------------------------------------
# 5. CONSULTAR VARIABLES REQUERIDAS
# ------------------------------------------------------------

@app.get("/features")
def get_features():

    return {
        "required_features": features
    }


# ------------------------------------------------------------
# 6. ENDPOINT DE PREDICCIÓN
# ------------------------------------------------------------

@app.post("/predict")
def predict_adr(request: PredictionRequest):

    try:

        prediction = predict(
            request.data
        )

        return {
            "predicted_adr": round(
                prediction,
                2
            )
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="No se pudo generar la predicción."
        )
