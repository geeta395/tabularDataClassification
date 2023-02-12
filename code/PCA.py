# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:08:02 2023

@author: chggo
"""
# libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.decomposition import PCA
import seaborn as sns
import os
import numpy as np



class PCAmethod():
    def __init__(self,dirc,point_color):
        self.dirc=dirc
        self.point_color=point_color
    
    def plotCorr(self,data):
    
        # Correlation 
        plt.figure(figsize=(10,10))
        # ticklabels = ['Age', 'E','Out-of-plane MCF angle', 'In-plane MCF angle','Mineral Crystallinity ratio', 
                        # 'Collagen cross-link ratio','Yield strain', 'Yield stress', 'BVTV', 'TMD']
        heatmap = sns.heatmap(data.corr(), square=False,
                              #xticklabels = ticklabels, yticklabels = ticklabels, 
                              cmap = "mako", linewidth=.5)
        heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=45) #cmap='plasma'
        heatmap.set(xlabel="", ylabel="")
        plt.savefig(os.path.join(self.dirc,'results','corr.png'))
        plt.close()
        #Note: Few pairs are highly correlated
        return
    
    
    def applyPCA(self,data,n_components=12):
        
        self.plotCorr(data)
        # Normalize data before applying PCA as PCA is highly sensitive to the variance of parameters
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        
        # PCA
        pca = PCA(n_components)
        pca.fit(data_normalized) 
        x_pca = pca.transform(data_normalized) 
        self.plotPCA(x_pca,pca,data)
        loading_scores = pd.Series(pca.components_[0], index=data.columns)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_parameters = sorted_loading_scores[0:10].index.values
        topPara=(loading_scores[top_parameters]).to_excel(os.path.join(self.dirc,'results','topPara.xlsx'))
        
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.vlines(x=10, ymax=1, ymin=0, colors="r", linestyles="--")
        plt.hlines(y=0.95, xmax=12, xmin=0, colors="g", linestyles="--")
        plt.title('PCA components to be used with 95% explained variance')
        plt.ylabel('cumulative sum')
        plt.xlabel('Total components')
        plt.plot(explained_variance)
        plt.savefig(os.path.join(self.dirc,'results','explained_Variance.png'))
        plt.close()
        return(x_pca)
    
    def plotPCA(self,x_pca,pca,data):
        # Plot PCA result
        ## color-coding the patients diagnoses
     
        plt.figure(figsize=(10,10))
        plt.scatter(x_pca[:,0],x_pca[:,1],s=40,c=self.point_color)
        plt.title('PCA Analysis')
        plt.xlabel('First Principal Component ({0:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
        plt.ylabel('Second Principal Component ({0:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))
        plt.savefig(os.path.join(self.dirc,'results','pca.png'))
        plt.close()
        return
       
       
    
