import numpy as np
import sys
from sklearn.decomposition import SparsePCA 


x_mat=np.loadtxt('in_data/news_train.txt')


DIM_Z=4

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

spca = SparsePCA(alpha=0.117,n_components=DIM_Z)
z_sp = spca.fit_transform(x_mat)
W_sp=spca.components_.T
print(str(np.sum(W_sp!=0))+ ' Connections, SP')
np.save('data/W_news_sp4',W_sp)
