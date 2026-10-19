"""Visualizaciones de exploración sin afirmar causalidad."""
import matplotlib.pyplot as plt
from pipeline import OUTPUT,load_data,prepare

def run(path=None):
    d=prepare(load_data(path) if path else load_data())
    print('Dimensión:',d.shape,'\nCancelación:',round(d.is_canceled.mean(),3))
    print('\nNulos:\n',d.isna().sum().sort_values(ascending=False).head(12))
    print('\nPor hotel:\n',d.groupby('hotel').agg(reservas=('hotel','size'),cancelacion=('is_canceled','mean'),adr_mediana=('adr','median')).round(2))
    (OUTPUT/'figures').mkdir(parents=True,exist_ok=True)
    ax=d.groupby(['hotel','arrival_month_num']).size().unstack(0).plot(figsize=(9,4),marker='o',title='Reservas por mes y tipo de hotel (2015–2017)')
    ax.set(xlabel='Mes de llegada',ylabel='Reservas');ax.figure.tight_layout();ax.figure.savefig(OUTPUT/'figures/seasonality.png',dpi=150);plt.close(ax.figure)
    ax=d.groupby('lead_time',observed=True).is_canceled.mean().rolling(30,min_periods=1).mean().plot(figsize=(8,4),title='Cancelación observada por anticipación (promedio móvil)')
    ax.set(xlabel='Lead time (días)',ylabel='Proporción cancelada');ax.figure.tight_layout();ax.figure.savefig(OUTPUT/'figures/cancellation_by_lead_time.png',dpi=150);plt.close(ax.figure)
    return d
if __name__=='__main__': run()
