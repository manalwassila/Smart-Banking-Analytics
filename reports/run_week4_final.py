import json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,fbeta_score,roc_auc_score,average_precision_score,confusion_matrix
X=pd.read_csv('data/processed/X_candidate.csv'); y=pd.read_csv('data/processed/y_target.csv')['TARGET'].astype(int)
cat=X.select_dtypes(include=['object','str']).columns.tolist(); num=[c for c in X if c not in cat]
X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.25,stratify=y_dev,random_state=43)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=44)
prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':(LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42),{'clf__C':[0.1,1,10]}),'Random Forest':(RandomForestClassifier(n_estimators=100,class_weight='balanced_subsample',n_jobs=-1,random_state=42),{'clf__min_samples_leaf':[3,8],'clf__max_features':['sqrt',0.8]})}
results={}
for name,(clf,grid) in models.items():
 pipe=Pipeline([('prep',prep),('clf',clf)])
 gs=GridSearchCV(pipe,grid,scoring='average_precision',cv=cv,n_jobs=1,refit=True)
 t=time.time(); gs.fit(X_train,y_train); elapsed=time.time()-t
 val_prob=gs.predict_proba(X_val)[:,1]
 thresholds=np.arange(.05,.801,.01)
 f2s=[fbeta_score(y_val,(val_prob>=th).astype(int),beta=2,zero_division=0) for th in thresholds]
 threshold=float(thresholds[int(np.argmax(f2s))])
 test_prob=gs.predict_proba(X_test)[:,1]; test_pred=(test_prob>=threshold).astype(int)
 tn,fp,fn,tp=confusion_matrix(y_test,test_pred,labels=[0,1]).ravel()
 results[name]={'cv_AP_mean':float(gs.best_score_),'params':gs.best_params_,'threshold':threshold,'validation_F2':float(max(f2s)),'test_accuracy':float(accuracy_score(y_test,test_pred)),'test_precision':float(precision_score(y_test,test_pred,zero_division=0)),'test_recall':float(recall_score(y_test,test_pred,zero_division=0)),'test_F1':float(f1_score(y_test,test_pred,zero_division=0)),'test_F2':float(fbeta_score(y_test,test_pred,beta=2,zero_division=0)),'test_ROC_AUC':float(roc_auc_score(y_test,test_prob)),'test_AP':float(average_precision_score(y_test,test_prob)),'confusion':{'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)},'seconds':round(elapsed,1)}
 print(name,json.dumps(results[name]),flush=True)
with open('reports/week4_final_experiments.json','w',encoding='utf-8') as f: json.dump({'n':len(X),'target_rate':float(y.mean()),'train_n':len(X_train),'validation_n':len(X_val),'test_n':len(X_test),'results':results},f,ensure_ascii=False,indent=2)
print('sizes',len(X_train),len(X_val),len(X_test),'target_rate',float(y.mean()),flush=True)
