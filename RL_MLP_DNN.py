#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:43:28 2020

@author: Paul Goyes-Peñafiel, Alejandra Hernandez-Rojas. (2020)
Con este script de python se realiza el cálculo y regresión o ajuste
de los modelos RL, MLP y DNN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

feature_cols = ['k1','k2','k3','k4','k5']
df = pd.read_csv('../SpreadSheet_PointsValues/400PointsValues.csv')
#Get the correlation matrix and graph. Figure XX

corrMatrix = df[feature_cols].corr()
plt.figure()
sn.heatmap(corrMatrix, annot=True)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Separating out the features
xx = df[feature_cols[2:]].values
k1k2=df[feature_cols[:2]].values
# Separating out the target
y = df["class"].values

# Standardizing the features
x = StandardScaler().fit_transform(xx)
x1 = StandardScaler().fit_transform(k1k2)
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1'])

print("The explained variance por 1 PC is: ", np.sum(pca.explained_variance_ratio_))

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)

X_LR = np.stack((finalDf.pc1.values,np.sum(x1,axis=1)),axis=1)

plt.figure(figsize = (8,8))
plt.subplot(1,2,1) 
plt.scatter(X_LR[:,0], X_LR[:,1], c=y,cmap="jet",edgecolors='k')
plt.colorbar()
## REGRESION LOGISTICA CON 2 COMPONENTES PRINCIPALES

from sklearn.linear_model import LogisticRegression
logistic= LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr',fit_intercept = True)

logistic.fit(X_LR,y)

#predict1=logistic.predict(X_LR)

res=100
_x0 = np.linspace(X_LR[:,0].min(), X_LR[:,0].max(), res)
_x1 = np.linspace(X_LR[:,1].min(), X_LR[:,1].max(), res)

XX,XY = np.meshgrid(_x0,_x1)
XXgridval=np.stack((XX.flatten(),XY.flatten()),1)

YYgridpred = logistic.predict_proba(XXgridval)[::,1]
YYgridpred=YYgridpred.reshape((res,res))

plt.figure()
plt.subplot(121)
plt.pcolormesh(_x0, _x1, YYgridpred, cmap="jet")
plt.colorbar()
plt.scatter(X_LR[:,0], X_LR[:,1], c=y,cmap="jet",edgecolors='k')

#How good is your model?
### calculate confusion matrix
from sklearn import metrics

plt.subplot(122)
y_pred_proba = logistic.predict_proba(X_LR)[::,1]
fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)
auc = metrics.roc_auc_score(y, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

## get summary information for logistic regression

import statsmodels.api as sm

X = pd.DataFrame(data=X_LR,index=None,columns=['A','B'])
X['intercept'] = 1.0

Y= finalDf['class'].copy();


logit1 = sm.Logit(Y,X)
#methods: bfgs lbfgs cg ncg minimize
logit1.fit(method='newton').summary()
logit1.fit(method='newton').summary2()
logit1.fit().params

### FIT THE MODEL WITH MACHINE LEARNING (MLP)

#import scipy.io as sio
import tensorflow.keras as kr


nn = [2,10,3,10,1]
model = kr.Sequential()
#model.add(kr.layers.Dense(nn[1],kernel_initializer='normal', activation='relu',input_dim=nn[0]))
#model.add(kr.layers.Dense(nn[-1],kernel_initializer='normal', activation='sigmoid'))
model.add(kr.layers.Dense(nn[1],kernel_initializer='normal', activation='relu',input_dim=nn[0]))
model.add(kr.layers.Dense(nn[2],kernel_initializer='normal', activation='relu'))
model.add(kr.layers.Dense(nn[3],kernel_initializer='normal', activation='relu'))
model.add(kr.layers.Dense(nn[4],kernel_initializer='normal', activation='sigmoid'))
loss= ["MSE", "binary_crossentropy","sparse_categorical_crossentropy",'mean_absolute_error']


lr = 0.001
model.compile(loss=loss[1], optimizer='adam', metrics=['acc'])
history=model.fit(X_LR,y.flatten(),epochs=100,validation_split = 0.25)
print(history.history.keys())
#hacer predicciones
res=100
_x0 = np.linspace(X_LR[:,0].min(), X_LR[:,0].max(), res)
_x1 = np.linspace(X_LR[:,1].min(), X_LR[:,1].max(), res)

XX,XY = np.meshgrid(_x0,_x1)
XXgridval=np.stack((XX.flatten(),XY.flatten()),1)

YYgridpred = model.predict(XXgridval)
YYgridpred=YYgridpred.reshape((res,res))

plt.subplot(121)
plt.pcolormesh(_x0, _x1, YYgridpred, cmap="jet")
plt.scatter(X_LR[:,0], X_LR[:,1], c=y,cmap="jet",edgecolors='k')
#plt.scatter(Xval[:,0], Xval[:,1], c=yval,cmap="jet",edgecolors='gray')
#plt.xlim(X_LR[:,0].min(), X_LR[:,0].max())
#plt.ylim(X_LR[:,1].min(), X_LR[:,1].max())
plt.colorbar()
ax1 = plt.subplot(122)
ax1.plot(history.history['acc'],'b',label='acc')
ax1.plot(history.history['val_acc'],'b+',label='val_acc')
ax1.set_ylabel('Accuracity', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()  
ax2.plot(history.history['loss'],'r',label='loss')
ax2.plot(history.history['val_loss'],'r+',label='val_loss')
ax2.set_ylabel('Loss function', color='r')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='r')
#plt.title(f'epoch:{epochs} Test accuracy {test_acc:0.3f}')

plt.subplot(122)
from sklearn import metrics
y_pred_proba = model.predict_proba(X_LR)[::1]
fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)
auc = metrics.roc_auc_score(y, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
