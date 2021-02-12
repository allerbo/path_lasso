import numpy as np
import os,sys
from sklearn.decomposition import SparsePCA, PCA
sys.path.insert(1,'..')
from helper_fcts import r2, best_match

NOISE=True
SP=True

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if NOISE:
  x_mat = np.loadtxt('in_data/gauss_train_n.txt')
  x_val = np.loadtxt('in_data/gauss_val_n.txt')
  ALPHA=0.6
else:
  x_mat = np.loadtxt('in_data/gauss_train.txt')
  x_val = np.loadtxt('in_data/gauss_val.txt')
  ALPHA=0.4

save_str = ''
if SP:
  pca=SparsePCA(alpha=ALPHA,n_components=2)
  save_str+='sp'
else:
  pca=PCA(n_components=2)
  save_str+='pca'

if NOISE:
  save_str+='_n'


x_mat = np.loadtxt('in_data/gauss_train_n.txt')
x_val = np.loadtxt('in_data/gauss_val_n.txt')
N_SAMPLES = x_mat.shape[0]
n_train = int(round(0.9 * N_SAMPLES))
r2s = []
conns_l = []
cms = []
np_seed=0
best_r2=-1

while len(r2s)<10:
  np.random.seed(np_seed)
  p = np.random.permutation(N_SAMPLES)
  x_train, x_test = x_mat[p][:n_train,:], x_mat[p][n_train:,:] 
  my_z = pca.fit_transform(x_train)
  W=pca.components_.T
  my_rec=np.matmul(np.matmul(x_val,W),W.T)
  if np.sum(W!=0)==4 or not SP:
    my_r2 = r2(x_val,my_rec)
    my_cm = best_match(my_rec, x_val)[0]
    r2s.append(my_r2)
    conns_l.append(np.sum(W!=0))
    cms.append(my_cm)
    if my_r2>best_r2:
      np.save('data/W_gauss_'+save_str,W)
  np_seed+=1

def mean_std_str(ll,sd=True):
  if sd:
    return '$%s' % float('%.2g' % np.mean(ll)) + '\\pm' + '%s$' % float('%.2g' % np.std(ll))
  return '$%s$' % float('%.2g' % np.mean(ll))

print(save_str + ' & ' + mean_std_str(conns_l)+ ' & '+ mean_std_str(r2s)+ ' & '+ mean_std_str(cms)+ '\\\\')
