"""Regresión de ADR observada: referencia de precios históricos, NO tarifa óptima causal."""
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from pipeline import ROOT,OUTPUT,load_data,prepare,feature_columns,split_data,make_preprocessor,save_json


def metrics(y,p):
    return {'mae':float(mean_absolute_error(y,p)),
            'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p))}


def run(path=None,temporal=False):
    df=prepare(load_data(path) if path else load_data())
    # Selección definida ANTES del split. Se estudian reservas NO canceladas para aproximar precio realizado.
    df=df[(df.is_canceled==0)&(df.adr>=0)&(df.adr<=1000)&(df.total_nights>0)].copy()
    train,val,test=split_data(df,temporal=temporal)
    cols=feature_columns(df,'regression')
    # 'is_canceled' eliminado de predictores: sería una etiqueta que solo conocemos después.
    models={'ridge':Ridge(alpha=10), 'random_forest':RandomForestRegressor(n_estimators=100,
                        max_depth=16,min_samples_leaf=5,n_jobs=-1,random_state=1)}
    fitted={}; report={}
    for name,model in models.items():
        p=Pipeline([('prep',make_preprocessor(train[cols])),('model',model)])
        p.fit(train[cols],train.adr)
        pred=p.predict(val[cols]);report[name]=metrics(val.adr,pred)
        fitted[name]=p
        print(name,{k:round(v,3) for k,v in report[name].items()})
    selected=min(report,key=lambda m:report[m]['mae'])
    import pandas as pd
    full=pd.concat([train,val]); final=Pipeline([('prep',make_preprocessor(full[cols])),('model',models[selected])])
    final.fit(full[cols],full.adr)
    pred=final.predict(test[cols]); report['selected_model']=selected
    report['test']=metrics(test.adr,pred)
    report['split']='temporal_arrival_proxy' if temporal else 'random_stratified_60_20_20'
    report['population']='non_canceled ADR 0..1000, positive nights; selection bias'
    save_json(report,'regression.json')
    (ROOT/'models').mkdir(exist_ok=True);joblib.dump({'pipeline':final,'columns':cols},ROOT/'models/adr_reference.joblib')
    (OUTPUT/'figures').mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(test.adr,pred,s=6,alpha=.15)
    ax.plot([0,1000],[0,1000],'k--');ax.set(xlim=(0,600),ylim=(0,600),xlabel='ADR real (u.m.)',ylabel='ADR predicha (u.m.)',title='Predicción de tarifa histórica');fig.tight_layout();fig.savefig(OUTPUT/'figures/adr_pred_vs_real.png',dpi=150);plt.close(fig)
    return report,final

if __name__=='__main__': run()
