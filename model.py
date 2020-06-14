# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# LOADING THE DATASET
df = pd.read_csv('diabetes.csv')

#SPLITTING THE 
y = df.Outcome
X = df.drop(labels='Outcome',axis=1)

#TRAIN TEST SPLIT AND STANDARD SCALER
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''x_train,x_test,y_train,y_test = train_test_split(xcopy,ycopy,test_size=0.2,random_state = 32)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''

#APPLYING FEW CLASSIFICATION TECHNIQUES
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,recall_score

dic = {'LR': LogisticRegression(max_iter=200),
       'KNN': KNeighborsClassifier(n_neighbors=10),
       'RF': RandomForestClassifier(),
       'DT':DecisionTreeClassifier(),
       'SVM': SVC(),
       }
k = list(dic.keys())

'''accuracy={}
recall={}
f1 = {}
accuracy_test={}
for i in k:
    model = dic[i]
    model.fit(x_train,y_train)
    y_pred = model.predict(x_train)
    y_t_pred = model.predict(x_test)
    ac = accuracy_score(y_true=y_train,y_pred=y_pred)
    accuracy[i] = ac
    recall[i] = recall_score(y_true=y_train,y_pred=y_pred)
    f1[i] = f1_score(y_true=y_train,y_pred=y_pred)
    accuracy_test[i] = accuracy_score(y_true=y_test,y_pred=y_t_pred)

'''
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=32)
X_tr = np.array(X_tr)
X_te = np.array(X_te)
y_tr = np.array(y_tr)
y_te = np.array(y_te)
accuracy_s={}
recall_s={}
f1_s = {}
accuracy_test_s={}
for it in k:
    model_s = dic[it]
    model_s.fit(X_tr,y_tr)
    y_pred_s = model_s.predict(X_tr)
    y_t_pred_s = model_s.predict(X_te)
    ac_s = accuracy_score(y_true=y_tr,y_pred=y_pred_s)
    accuracy_s[it] = ac_s
    recall_s[it] = recall_score(y_true=y_tr,y_pred=y_pred_s)
    f1_s[it] = f1_score(y_true=y_tr,y_pred=y_pred_s)
    accuracy_test_s[it] = accuracy_score(y_true=y_te,y_pred=y_t_pred_s)
    
opt_model = dic['SVM']
pickle.dump(opt_model,open('model.pkl','wb'))

#zz = pickle.load(open('model.pkl','rb'))
#rint(zz.predict([[0,35,72,35,100,32.5,0.5,50]]))

