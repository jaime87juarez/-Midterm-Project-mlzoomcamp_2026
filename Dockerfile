# ============================================================
# HOTEL YIELD ML
# DOCKERFILE PARA FASTAPI
# ============================================================

# 1. Imagen base de Python
FROM python:3.11-slim


# 2. Directorio de trabajo
WORKDIR /app


# 3. Copiar dependencias
COPY requirements.txt .


# 4. Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt


# 5. Copiar aplicación FastAPI
COPY app/ ./app/


# 6. Copiar modelo entrenado
COPY models/ ./models/


# 7. Puerto utilizado por FastAPI
EXPOSE 8000


# 8. Ejecutar servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
