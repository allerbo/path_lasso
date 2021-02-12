import numpy as np
import sys,os
sys.path.insert(1,'..')
from path_pen import path_pen

x_mat=np.loadtxt('in_data/news_train.txt')

OPT_SEED=1

LBDA_L1=0.003
ZERO_THRESH=.25

DIM_Z=2

epochs1=100000
epochs2=100000
epochs3=100000
epochs4=100000
dim_h=50

LOG_NAME = 'news_l12'

#Train with l1 penalty until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs1, LBDA_L1=LBDA_L1, LOG_NAME=LOG_NAME, START_BEST=2000)
os.system('bash ../copy_params.sh phase0_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Threshold and train without penalty with fixed zeros until convergennce
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase0_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=DIM_Z, EPOCHS=epochs2, ZERO_THRESH=ZERO_THRESH, LOG_NAME=LOG_NAME, START_BEST=2000)
os.system('bash ../copy_params.sh phase4_'+LOG_NAME+'_best '+LOG_NAME+'_best')

