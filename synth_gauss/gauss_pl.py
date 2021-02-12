import numpy as np
import os,sys
sys.path.insert(1,'..')
from path_pen import path_pen

epochs1=100000
epochs2=100000
epochs3=100000
epochs4=100000
dim_h=50
dim_z=2

LBDA_PATH=.01
LBDA_EXCL=.01

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if NOISE:
  x_mat = np.loadtxt('in_data/gauss_train_n.txt')
  n='n_'
  NP_SEED=2
  OPT_SEED=3
else:
  x_mat = np.loadtxt('in_data/gauss_train.txt')
  n=''
  NP_SEED=6
  OPT_SEED=1


LOG_NAME='gauss_pl_'+n+str(NP_SEED)+'_'+str(OPT_SEED)

#Train with exclusive lasso until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs1, LBDA_EXCL=LBDA_EXCL, START_BEST=1000, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase0_'+LOG_NAME+'_best '+LOG_NAME+'_best')
 
#Train with path lasso until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase0_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs2, LBDA_PATH=LBDA_PATH, START_BEST=1000, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase1_'+LOG_NAME+'_best '+LOG_NAME+'_best')
 
#Train with path lasso and proximal gradient descent until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs3, LBDA_PATH_PROX=1e-3, STEP_SIZE=1e-2, START_BEST=15, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase2_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Train without penalty with fixed zeros until convergennce
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase2_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs4, ZERO_THRESH=0, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase3_'+LOG_NAME+'_best '+LOG_NAME+'_best')
