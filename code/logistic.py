# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:49:34 2023

@author: chggo
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import os
from PCA import PCAmethod
from operator import itemgetter

class LogistcReg():
    def __init__(self,dirc):
        self.dirc=dirc

    def readData(self,dirc):
        df = pd.read_excel(os.path.join(dirc,'Femhals_all_mean_wpatientinfo_MLclustering.xlsx'),header=0)
        df = pd.get_dummies( df ,drop_first=True )
        df = df.dropna()
        
        y = df.Fracture_yes
        x = df.drop('Fracture_yes' , axis=1)
        return(df,x,list(y))
    
    
    def calc_vif(self,df):
        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(df)
        vif = pd.DataFrame()
        vif["variables"] = df.columns
        vif["VIF"] = [variance_inflation_factor(df_normalized, i) for i in range(df_normalized .shape[1])]
        return(vif)
    
    def removeMultiCollinearity(self,n_components):
       
        df,x,y=self.readData(self.dirc)
       
        vif=self.calc_vif(df).sort_values('VIF', ascending=False)
        #manually dropping variables one by one with excessive collinearity
        x_reduced = x.drop(['BMD','E_mean','Mineral2Matrix_mean'], axis=1)
        #x_reduced=x
        final_vif=(self.calc_vif(x_reduced).sort_values('VIF', ascending=False)).to_excel(os.path.join(self.dirc,'results','VIF.xlsx'))
        
        # apply PCA
        point_color=["lightgray" if y[idx]==0 else "darkcyan" for idx in range(len(y))]
        P=PCAmethod(self.dirc,point_color)
        x_pca=P.applyPCA(x_reduced,n_components)
        return(x_pca,y,x_reduced)
    
    def predict(self,method,n_components=11,kernel='linear',n_neighbors=5, metric='euclidean'):
        
        x_pca,y,x=self.removeMultiCollinearity(n_components)
        x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.25,random_state=7) 
        
        if method=='Logistic':
            model=self.Logistic(x_train, y_train)
        if method=='svm':
            model=self.SVM(x_train, y_train,kernel)
        if method=='naive':
            model=self.naiveByes(x_train, y_train)
        if method=='k_near':
            model=self.K_nearest(x_train, y_train,n_neighbors, metric)
        
            
        y_pred= model.predict(x_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        self.confusionMatrix(cm,method)
        report=metrics.classification_report(y_test,y_pred)
        score = model.score(x_test, y_test)
        
        if method != 'naive' and method != 'k_near':
            w=model.coef_[0]#get the weights
            sorted_features=sorted([[x.columns[i],abs(w[i])] for i in range(len(w))],key=itemgetter(1),reverse=True)
        else:
            sorted_features=None


        return(score,y_pred,y_test,sorted_features)
        
    def Logistic(self,x_train, y_train):
       
        LR=LogisticRegression(solver='liblinear',C=10.0) 
        model=LR.fit(x_train, y_train)
        return(model)
    
    def SVM(self,x_train, y_train,kernel):
       
        model=svm.SVC(kernel=kernel)
        model.fit(x_train, y_train)
        return(model)
    
    def naiveByes(self,x_train, y_train):
        model = GaussianNB()
        model.fit(x_train, y_train)
        return(model)
    
    def K_nearest(self,x_train, y_train,n_neighbors, metric):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        model.fit(x_train, y_train)
        return(model)
    
    def confusionMatrix(self,cm,method):
        
        cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = ['Coxarthrosis','Fracture'])
        cm_display.plot()
        plt.savefig(os.path.join(self.dirc,'results',method+ '-'+'confusioMatrix.png'))
        plt.close()
        return
        


