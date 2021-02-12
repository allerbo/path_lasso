import numpy as np
import itertools as it

def nor(z):
  return (z-np.mean(z,0))/np.std(z,0)

N_X_CL=100
DIM=4
np.random.seed(0)
sigma=np.zeros((DIM,DIM))
np.fill_diagonal(sigma,1)
sigma=0.01*sigma

mus = list(it.product([0,1],repeat=DIM))

x_mat = np.empty((0,DIM))
lab=[]
for cl in range(len(mus)):
  x_mat=np.vstack((x_mat,np.random.multivariate_normal(mus[cl],sigma,N_X_CL)))
  lab+=N_X_CL*[cl]

lab=np.array(lab)

x_mat_n = np.copy(x_mat)

x_mat_n+= np.random.normal(0,0.3,x_mat.shape)

x_mat = nor(x_mat)
x_mat_n = nor(x_mat_n)
N_X=x_mat.shape[0]
n_train = int(round(0.8 * N_X))
p = np.random.permutation(N_X)
x_train, x_val = x_mat[p][:n_train,:], x_mat[p][n_train:,:] 
x_train_n, x_val_n = x_mat_n[p][:n_train,:], x_mat_n[p][n_train:,:] 
lab_train, lab_val = lab[p][:n_train], lab[p][n_train:] 
np.savetxt('in_data/gauss_train.txt',x_train)
np.savetxt('in_data/gauss_val.txt',x_val)
np.savetxt('in_data/gauss_train_n.txt',x_train_n)
np.savetxt('in_data/gauss_val_n.txt',x_val_n)
np.savetxt('in_data/gauss_lab_train.txt',lab_train)
np.savetxt('in_data/gauss_lab_val.txt',lab_val)
