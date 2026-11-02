# ============================================================
# HOTEL YIELD ML
# DEPLOYMENT DEL MODELO RANDOM FOREST CON FASTAPI
# ============================================================

# 1. Imagen base de Python
FROM python:3.11-slim

# 2. Directorio principal del contenedor
WORKDIR /app

# 3. Evitar archivos innecesarios de Python
ENV PYTHONDONTWRITEBYTECODE=1

# 4. Mostrar mensajes de Python inmediatamente
ENV PYTHONUNBUFFERED=1

# 5. Copiar dependencias
COPY requirements-docker.txt .

# 6. Instalar las bibliotecas
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

# 7. Copiar aplicación
COPY app/ ./app/

# 8. Copiar modelo entrenado
COPY models/ ./models/

# 9. Puerto utilizado por FastAPI
EXPOSE 8000

# 10. Iniciar servidor
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
