"""Pipeline didáctico: separación, imputación y codificación sin fuga de información."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'hotel_bookings.csv'
OUTPUT = ROOT / 'outputs'
MONTHS = {m: i for i, m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
# Evitar datos que solo se conocen después de alojarse, cancelar o gestionar la reserva.
POST_BOOKING = ['reservation_status','reservation_status_date','assigned_room_type',
                'booking_changes','days_in_waiting_list','required_car_parking_spaces',
                'total_of_special_requests']
# ADR es conocido como tarifa acordada, pero NO se debe usar para predecirse a sí mismo.
ID_FEATURES = ['agent','company']


def load_data(path=DATA):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'No se encontró {path}. Ejecuta python src/download_data.py o copia el CSV de Kaggle.')
    df = pd.read_csv(path, low_memory=False)
    expected = {'hotel','is_canceled','adr','lead_time','arrival_date_year','arrival_date_month'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {missing}. Se espera hotel_bookings.csv de Kaggle.')
    return df


def prepare(df):
    """Transformaciones deterministas antes del split (nunca aprender medias aquí)."""
    x = df.copy()
    x['arrival_month_num'] = x['arrival_date_month'].map(MONTHS)
    x['arrival_date'] = pd.to_datetime(dict(year=x.arrival_date_year,
                     month=x.arrival_month_num, day=x.arrival_date_day_of_month), errors='coerce')
    x['total_nights'] = x.stays_in_weekend_nights + x.stays_in_week_nights
    x['total_guests'] = x.adults + x.children.fillna(0) + x.babies
    return x
