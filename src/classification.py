
"""Clasificación: regresión logística, Random Forest y optimización por costo esperado."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score,precision_score,recall_score,f1_score,
                             accuracy_score,confusion_matrix,RocCurveDisplay,
                             precision_recall_curve,average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pipeline import ROOT,OUTPUT,load_data,prepare,feature_columns,split_data,make_preprocessor,save_json


def metrics(y, probs, threshold=.5):
    pred = (np.asarray(probs)>=threshold).astype(int)
    return dict(auc=float(roc_auc_score(y,probs)),ap=float(average_precision_score(y,probs)),
                accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
                threshold=float(threshold),confusion_matrix=confusion_matrix(y,pred).tolist())


def threshold_table(y, p, contact_cost=3., recovered_revenue=90., false_alarm_cost=12.):
    """Escenario ILUSTRATIVO: se contacta a reservas señaladas; NO demuestra ingresos reales."""
    rows=[]
    for t in np.round(np.arange(.05,.96,.01),2):
        pred=np.asarray(p)>=t
        tp=int(((np.asarray(y)==1)&pred).sum())
        fp=int(((np.asarray(y)==0)&pred).sum())
        # solo ejemplo: beneficio esperado = TP * recuperación supuesta - costos de contacto y molestias
        utility=tp*recovered_revenue - int(pred.sum())*contact_cost - fp*false_alarm_cost
        rows.append((float(t),float(f1_score(y,pred,zero_division=0)),tp,fp,float(utility)))
    return pd.DataFrame(rows,columns=['threshold','f1','true_positives','false_positives','utility_proxy_usd'])


def run(path=None, temporal=False):
    raw=load_data(path) if path else load_data()
    df=prepare(raw)
    train,val,test=split_data(df,temporal=temporal)
    cols=feature_columns(df,'classification')
    X_train,X_val,X_test=(d[cols] for d in (train,val,test))
    y_train,y_val,y_test=(d.is_canceled.astype(int) for d in (train,val,test))
    models={
       'logistic':LogisticRegression(solver='liblinear',C=1.0,max_iter=1000),
       'random_forest':RandomForestClassifier(n_estimators=120,min_samples_leaf=8,
                                          random_state=1,n_jobs=-1,max_features='sqrt')}
    fitted={}; report={}
    for name,est in models.items():
        pipe=Pipeline([('prep',make_preprocessor(X_train)),('model',est)])
        pipe.fit(X_train,y_train)
        probs=pipe.predict_proba(X_val)[:,1]
        report[name]=metrics(y_val,probs)
        fitted[name]=pipe
        print(name, {k:round(report[name][k],3) for k in ('auc','ap','precision','recall','f1')})
    chosen=max(report,key=lambda m:report[m]['auc'])
    p_val=fitted[chosen].predict_proba(X_val)[:,1]
    curve=threshold_table(y_val,p_val)
    best_f1=curve.loc[curve.f1.idxmax()]
    best_utility=curve.loc[curve.utility_proxy_usd.idxmax()]
    # Reconstruir a partir de train+val: elegir modelo/umbral SOLO con validación.
    combined=pd.concat([train,val])
    final=Pipeline([('prep',make_preprocessor(combined[cols])),('model',models[chosen])])
    final.fit(combined[cols],combined.is_canceled.astype(int))
    p_test=final.predict_proba(X_test)[:,1]
    report['selected_model']=chosen
    report['threshold_max_f1']=float(best_f1.threshold)
    report['threshold_utility_illustrative']=float(best_utility.threshold)
    report['test']=metrics(y_test,p_test,float(best_f1.threshold))
    report['split']='temporal_arrival_proxy' if temporal else 'random_stratified_60_20_20'
    save_json(report,'classification.json')
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump({'pipeline':final,'columns':cols,'threshold':float(best_f1.threshold)},ROOT/'models/cancellation.joblib')
    (OUTPUT/'figures').mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(curve.threshold,curve.f1,label='F1');ax.axvline(best_f1.threshold,color='gray',ls='--',label='Máx. F1');ax.set(xlabel='Umbral',ylabel='F1',title='Selección de umbral (validación)');ax.legend();fig.tight_layout();fig.savefig(OUTPUT/'figures/threshold_f1.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));RocCurveDisplay.from_predictions(y_test,p_test,ax=ax);fig.tight_layout();fig.savefig(OUTPUT/'figures/roc_classification.png',dpi=150);plt.close(fig)
    return report,curve,final

if __name__=='__main__':
    run()
