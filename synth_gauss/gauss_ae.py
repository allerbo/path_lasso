import numpy as np
import os,sys
sys.path.insert(1,'..')
from path_pen import path_pen

epochs1=100000
dim_h=50
dim_z=2

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if NOISE:
  x_mat = np.loadtxt('in_data/gauss_train_n.txt')
  n='n_'
  NP_SEED=5
  OPT_SEED=2
else:
  x_mat = np.loadtxt('in_data/gauss_train.txt')
  n=''
  NP_SEED=4
  OPT_SEED=1

LOG_NAME='gauss_ae_'+n+str(NP_SEED)+'_'+str(OPT_SEED)

#Train without penalty until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs1, START_BEST=100, LOG_NAME=LOG_NAME, PRINT=True)
os.system('bash ../copy_params.sh phase0_'+LOG_NAME+'_best '+LOG_NAME+'_best')
