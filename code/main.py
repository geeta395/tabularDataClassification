# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:03:48 2023

@author: Dell
"""

from logistic import LogistcReg
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# path
dirc=r'C:\Users\Dell\Downloads\Femhals_shared_w_Geeta\Femhals_shared_w_Geeta'

# Call file
L=LogistcReg(dirc)

# create result folder
try: 
    os.mkdir(os.path.join(dirc,'results')) 
except OSError as error: 
    print(error)  

# methods
methods=['Logistic','svm','naive','k_near']

def plotPred(y_pred,y_test,method):
    X=np.arange(0,len(y_pred))
    plt.scatter(X,y_pred,color='green',label='predicted')
    plt.scatter(X,y_test,color='blue',label='test')
    plt.legend()
    plt.title(method+'-'+'Prediction')
    plt.ylabel('labels')
    plt.xlabel('predicted/true label')
    plt.savefig(os.path.join(dirc,'results', method + '-' + 'Prediction.png'))
    plt.close()
    return


result=[]
for i in methods:
    score,y_pred,y_test,sorted_features=L.predict(i)
    result.append({'method':i,'score':score,'y_pred':y_pred,'y_test':y_test,'features':sorted_features})
    plotPred(y_pred,y_test,i)

result=pd.DataFrame(result)
result.to_excel(os.path.join(dirc,'results', 'finalResult.xlsx'))