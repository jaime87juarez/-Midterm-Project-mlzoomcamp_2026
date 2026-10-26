#load_data,prepare,split_data,feature_columns,make_preprocessor

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


def feature_columns(df, task):
    """Variables al momento de reservar, comunes a ambas tareas."""
    omitted = set(POST_BOOKING + ID_FEATURES + ['is_canceled','adr','arrival_date',
        'arrival_date_month','arrival_date_week_number','arrival_date_day_of_month',
        'arrival_date_year','country'])
    # country se excluye por cardinalidad y para evitar dependencia de atributos sensibles de origen.
    cols = [c for c in df.columns if c not in omitted]
    cols = [c for c in cols if c not in ('total_nights', 'total_guests')] + ['total_nights','total_guests','arrival_month_num']
    return list(dict.fromkeys(c for c in cols if c in df.columns))


def split_data(df, temporal=False):
    if temporal:
        ordered = df.sort_values('arrival_date').reset_index(drop=True)
        n = len(ordered)
        return ordered.iloc[:int(.6*n)].copy(), ordered.iloc[int(.6*n):int(.8*n)].copy(), ordered.iloc[int(.8*n):].copy()
    full, test = train_test_split(df, test_size=.2, random_state=1, stratify=df.is_canceled)
    train, val = train_test_split(full, test_size=.25, random_state=1, stratify=full.is_canceled)
    return train.copy(),val.copy(),test.copy()


def make_preprocessor(X):
    cat = list(X.select_dtypes(include=['object','category','string']).columns)
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('num',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler())]),num),
        ('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='NA')),
                          ('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ], remainder='drop')


def save_json(data, name):
    path=OUTPUT/'metrics'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,)): return float(x)
        raise TypeError(str(type(x)))
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=convert),encoding='utf8')
    return path
