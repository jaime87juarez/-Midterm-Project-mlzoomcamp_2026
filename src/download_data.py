"""Descarga oficial opcional (requiere conexión). Alternativa: descargar manualmente Kaggle."""
from pathlib import Path
from urllib.request import urlretrieve
from pipeline import DATA

URL='https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv'
if __name__=='__main__':
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        print(f'Ya existe: {DATA}')
    else:
        print('Descargando copia pública TidyTuesday del conjunto Hotel Booking Demand...')
        try:
            urlretrieve(URL, DATA)
            import pandas as pd
            print(f'Archivo listo: {DATA} / {len(pd.read_csv(DATA)):,} filas')
        except Exception as exc:
            DATA.unlink(missing_ok=True)
            print(f'Error descargando: {exc}\nDescarga hotel_bookings.csv de Kaggle y colócalo en data/raw/.')
            raise SystemExit(1)
