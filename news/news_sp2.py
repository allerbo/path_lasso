import numpy as np
from sklearn.decomposition import SparsePCA 
import sys,os,datetime,re


x_mat=np.loadtxt('in_data/news_train.txt')

DIM_Z=2

def nor(z):
  return (z-np.mean(z,0))/np.std(z,0)
spca = SparsePCA(alpha=.25,n_components=DIM_Z)
z_sp = spca.fit_transform(x_mat)
W_sp=spca.components_.T
print(str(np.sum(W_sp!=0))+ ' Connections, SP')
np.save('data/W_news_sp2',W_sp)
