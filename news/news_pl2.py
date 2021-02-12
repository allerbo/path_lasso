import numpy as np
import sys,os
sys.path.insert(1,'..')
from path_pen import path_pen


x_mat=np.loadtxt('in_data/news_train.txt')

OPT_SEED=1

LBDA_PATH=.0007
LBDA_EXCL=.0003

DIM_Z=2

epochs1=100000
epochs2=100000
epochs3=100000
epochs4=100000
dim_h=50

LOG_NAME = 'news_pl2'
SAVE_NAME=LOG_NAME

#Train without penalty until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs1, LOG_NAME=LOG_NAME, START_BEST=2000)
os.system('bash ../copy_params.sh phase0_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Train with exclusive lasso until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase0_'+SAVE_NAME+'_best', DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs1, LOG_NAME=LOG_NAME, LBDA_EXCL=LBDA_EXCL, START_BEST=2000)
os.system('bash ../copy_params.sh phase1_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Train with path lasso until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+SAVE_NAME+'_best', DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs2, LBDA_PATH=LBDA_PATH, START_BEST=100, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase2_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Train with path lasso and proximal gradient descent until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase2_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs3, LBDA_PATH_PROX=2e-7, STEP_SIZE=0.5, START_BEST=15, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase3_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Train without penalty with fixed zeros until convergennce
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase3_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs4, ZERO_THRESH=0, LOG_NAME=LOG_NAME)
os.system('bash ../copy_params.sh phase4_'+LOG_NAME+'_best '+LOG_NAME+'_best')
