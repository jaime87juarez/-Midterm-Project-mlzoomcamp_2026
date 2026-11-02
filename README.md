# Hotel Yield ML — ML Zoomcamp 2026 Midterm Project

**Author:** Jaime Juárez  
**Course:** Machine Learning Zoomcamp, 2026 cohort  
**Task:** Predict the Average Daily Rate (ADR) of a hotel booking and deploy one selected model using FastAPI and Docker.

## Contents

1. [Project overview](#1-project-overview)
2. [Dataset and problem definition](#2-dataset-and-problem-definition)
3. [Exploratory analysis and modeling](#3-exploratory-analysis-and-modeling)
4. [Model selection and results](#4-model-selection-and-results)
5. [Repository structure](#5-repository-structure)
6. [Reproduce the development environment](#6-reproduce-the-development-environment)
7. [Run the notebooks and train the final model](#7-run-the-notebooks-and-train-the-final-model)
8. [Run the API locally](#8-run-the-api-locally)
9. [Deploy with Docker](#9-deploy-with-docker)
10. [API reference and example prediction](#10-api-reference-and-example-prediction)
11. [Testing and troubleshooting](#11-testing-and-troubleshooting)
12. [Limitations and next steps](#12-limitations-and-next-steps)

## 1. Project overview

Hotel rooms are **perishable inventory**: a room-night that goes unsold cannot be recovered after that date. Hotel managers must balance price, occupancy, seasonality, booking lead time, distribution channels, and cancellation risk.

This project applies supervised machine learning to historical booking records. Its **deployed task is ADR regression**:

> Given the available attributes of a hotel booking, what daily room rate would a model predict from historical bookings?

The repository includes exploratory data analysis (EDA), a separate cancellation-classification exercise, regression model comparison, and an HTTP inference API.

**Scope boundary:** The deployed model predicts *historically observed booked ADR*. It is **not** a price optimizer, an estimate of the causal effect of price on demand, or a forecast of total market demand.

### Deployment architecture

```text
Hotel Booking Demand dataset
            |
            v
  src/pipeline.py (preparation)
            |
            v
   train.py (one final model)
            |
            v
 models/final_model.joblib
   [preprocessing + Random Forest]
            |
            v
    app/predic.py (inference)
            |
            v
      app/main.py (FastAPI)
            |
            v
      Docker container
            |
            v
      POST /predict → ADR
```

## 2. Dataset and problem definition

**Source:** [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). The widely used dataset contains approximately **119,390 bookings and 32 original fields** for a City Hotel and a Resort Hotel, covering arrivals from **July 2015 through August 2017**.

Examples of useful attributes include booking lead time, length of stay, guest counts, market segment, distribution channel, room type, and arrival month.

### Regression target and filtering

- **Target:** `adr` (Average Daily Rate).
- **Selected records:** `is_canceled == 0`, `0 <= adr <= 1000`, and `total_nights > 0`.
- **Derived predictors:** The preprocessing pipeline adds such fields as `arrival_month_num`, `total_nights`, and `total_guests`.

These filters define the modeling population; they do not reveal the bookings that might have occurred at other prices. ADR predictions should not be interpreted as recommended selling prices.

### Companion classification task

`notebooks/02_Clasificacion.ipynb` explores cancellation prediction (`is_canceled`). Classification can help assess cancellation risk; it is **not** the model deployed in this project. Possible classification metrics include precision, recall, F1-score, and ROC-AUC.

## 3. Exploratory analysis and modeling

The analysis uses the following workflow:

1. Load and inspect booking data; examine nulls, categories, and distributions.
2. Explore seasonality, ADR, lead time, booking segments, and cancellations.
3. Prepare the regression dataset and split it into training, validation, and test sets using the project pipeline.
4. Fit preprocessing **inside** a Scikit-learn `Pipeline` on the training data.
5. Compare a regularized linear baseline (**Ridge**) with a nonlinear ensemble (**Random Forest**).
6. Select a single model using **validation** performance, then reserve the test set for a final unbiased evaluation.
7. Serialize the selected fitted pipeline and expose it for inference.

**Why the pipeline matters:** Encoding and imputation learned during training must also be applied at prediction time. The serialized artifact includes these fitted transformations, not only the regressor.

## 4. Model selection and results

### Validation-set scores

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Ridge Regression | 26.425 | 36.309 | 0.432 |
| **Random Forest Regressor** | **14.324** | **22.572** | **0.780** |

Random Forest had the lowest mean absolute error and root mean squared error and the highest R² on the reported validation split. Its MAE was about **45.8% lower** than Ridge's.

**Important:** These are **validation** results, not final held-out test scores. Add test-set results separately after running the selected model on the untouched test set. Do not adjust hyperparameters against test performance.

### Selected model

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=5,
    random_state=1,
    n_jobs=-1,
)
```

Only **one model** is deployed. `train.py` fits and saves the selected Random Forest pipeline; `app/predic.py` loads that saved artifact and predicts ADR. Ridge remains a benchmark in the notebook, not a second deployed service.

## 5. Repository structure

```text
HOTEL_YIELD_ML/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application and routes
│   └── predic.py                # Model loading and prediction logic
├── docs/
│   ├── DICCIONARIO.md
│   ├── GUIA_DIDACTICA.md
│   ├── indicaciones_venv.md
│   └── METODOLOGIA.md
├── models/
│   └── final_model.joblib       # Generated by train.py;
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Clasificacion.ipynb
│   └── 03_Regresion_Yield.ipynb
├── src/
│   ├── classification.py
│   ├── download_data.py
│   ├── eda.py
│   ├── pipeline.py
│   └── regression.py
├── tests/
├── train.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt           # Development / notebooks / training
├── requirements-docker.txt    # Lean inference dependencies
└── README.md
```

> The tree describes the intended workflow. Generated artifacts and the downloaded dataset may be excluded from version control. For reproducibility, ensure the dataset can be obtained and `train.py` can recreate `models/final_model.joblib`.

## 6. Reproduce the development environment

### Prerequisites

- **Python 3.11**, preferably matching the interpreter and packages used to train the saved artifact.
- Git and internet access to install dependencies / obtain the Kaggle dataset.
- Docker Desktop or Docker Engine **only** if running the containerized API.
- VS Code with Python/Jupyter extensions (optional, for notebooks).

> **Dependency files:** `requirements.txt` contains the project's **direct development dependencies** (data science, Jupyter, model training, API, and testing). `requirements-docker.txt` installs the smaller set needed to **serve** the trained model.

## 7. Run the notebooks and train the final model



### Recommended notebook order

1. `notebooks/01_EDA.ipynb` — explore booking data and pricing patterns.
2. `notebooks/02_Clasificacion.ipynb` — experiment with cancellation classification (not deployed).
3. `notebooks/03_Regresion_Yield.ipynb` — compare ADR regression models and select Random Forest.

### Train one final model

From the repository root, after providing the dataset at the location expected by the pipeline:

```bash
python train.py
```

Expected artifact:

```text
models/final_model.joblib
```

It contains the fitted preprocessing + Random Forest pipeline, feature names, and model metadata used by the prediction service. **Training and serving are separate:** API requests load the existing artifact; they do not retrain it.

## 8. Run the API locally

Make sure `models/final_model.joblib` exists, and install the development dependencies as described above. From the repository root:

```bash
python -m uvicorn app.main:app --reload
```

Open:

- Swagger UI: <http://localhost:8000/docs>
- Root status: <http://localhost:8000/>
- Health endpoint: <http://localhost:8000/health>
- Required features: <http://localhost:8000/features>

Stop the local server with `Ctrl+C` before running Docker on the same port.

## 9. Deploy with Docker

Docker serves the **same one saved model**. It does not need notebooks or training data at runtime.

### Build the image

Ensure `models/final_model.joblib` exists, then run from the root of the repository:

```bash
docker build -t hotel-yield-ml:v1 .
```

The `Dockerfile` installs `requirements-docker.txt`, copies `app/` and `models/`, and starts Uvicorn on port 8000.

### Run and inspect

```bash
docker run -d --name hotel-yield-api -p 8000:8000 hotel-yield-ml:v1
docker ps
docker logs hotel-yield-api
```

Then test <http://localhost:8000/docs>. To stop and remove the container:

```bash
docker stop hotel-yield-api
docker rm hotel-yield-api
```

### Rebuild after updating the code, dependencies, or model

```bash
docker build -t hotel-yield-ml:v1 .
docker rm -f hotel-yield-api
docker run -d --name hotel-yield-api -p 8000:8000 hotel-yield-ml:v1
```


## 10. API reference and example prediction

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome message and running status |
| `GET` | `/health` | Service health response |
| `GET` | `/features` | Names of features required by the model |
| `POST` | `/predict` | Predict ADR for one booking |

### Input features

`GET /features` returns **19 required fields**:

```json
{
  "required_features": [
    "hotel", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "meal", "market_segment",
    "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "deposit_type",
    "customer_type", "arrival_month_num", "total_nights", "total_guests"
  ]
}
```

### Example `POST /predict` request

The API expects the features **inside a top-level `data` object**:

```json
{
  "data": {
    "hotel": "City Hotel",
    "lead_time": 30,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "arrival_month_num": 7,
    "total_nights": 5,
    "total_guests": 2
  }
}
```

**Observed response from the working API:**

```json
{
  "predicted_adr": 130
}
```

The service predicted an ADR of **130 per night** for this sample booking. For five nights, that corresponds to **650 in modeled room revenue**, assuming the ADR applies to all five room-nights and ignoring taxes and extras. The response depends on the supplied input, saved artifact, and dependency environment. It does **not** identify a revenue-maximizing price.

### Test through Swagger

Open <http://localhost:8000/docs> → `POST /predict` → **Try it out** → replace the request body with the complete JSON above → **Execute**. A successful response has HTTP status `200` and a numeric `predicted_adr`.

### Test through PowerShell

```powershell
$body = @'
{
  "data": {
    "hotel": "City Hotel",
    "lead_time": 30,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "arrival_month_num": 7,
    "total_nights": 5,
    "total_guests": 2
  }
}
'@

Invoke-RestMethod -Uri 'http://localhost:8000/predict' `
  -Method Post -ContentType 'application/json' -Body $body
```

## 11. Testing and troubleshooting

### Basic checks

```bash
python -m pytest tests/ -v
```

Automated tests depend on the assertions actually implemented under `tests/`. The following additional smoke checks are recommended:

- `GET /` returns `"status": "running"`.
- `GET /features` returns the 19 fields listed above.
- `POST /predict` with the sample JSON returns HTTP `200` and a numeric ADR.
- The local and Docker API produce matching outputs with the same model artifact and inputs.
- Requests with missing/invalid input produce informative client errors.

### Common issues

## 12. Limitations and next steps

- **Not dynamic pricing:** Predicting previously observed ADR does not calculate the optimal tariff.
- **Selection bias:** Restricting the regression sample to non-cancelled bookings excludes other demand outcomes.
- **No causal inference:** Historical associations do not establish how customers will react to changing prices.
- **Data leakage risk:** Every input feature must be available at prediction time; review the pipeline and source fields accordingly.
- **Model evaluation:** Publish held-out test MAE, RMSE, and R² before claiming generalization quality.
- **Operational readiness:** Consider explicit Pydantic input validation, structured logging, monitoring, model versioning, and automated API tests.
- **Revenue management extension:** Modeling booking demand at different offered prices, capacity constraints, cancellations, and seasonality would be required before attempting price optimization.

---

### Project note

This project was prepared for the **ML Zoomcamp 2026** midterm assignment. AI assistance was used to help organize and polish parts of the README; model development, testing, and deployment should be verified against the repository and reproducible commands.
