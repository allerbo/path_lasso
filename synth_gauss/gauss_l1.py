import numpy as np
import os,sys
sys.path.insert(1,'..')
from path_pen import path_pen

def count_conns(LOAD_APDX, ZERO_THRESH):
  import numpy as np
  W1 = np.load('data/W1_'+LOAD_APDX+'.npy')
  W2 = np.load('data/W2_'+LOAD_APDX+'.npy')
  W3 = np.load('data/W3_'+LOAD_APDX+'.npy')
  W4 = np.load('data/W4_'+LOAD_APDX+'.npy')
  W1_01 = 1*(np.abs(W1)>ZERO_THRESH)
  W2_01 = 1*(np.abs(W2)>ZERO_THRESH)
  W3_01 = 1*(np.abs(W3)>ZERO_THRESH)
  W4_01 = 1*(np.abs(W4)>ZERO_THRESH)
  W_GL_01=np.matmul(W1_01,W2_01)+np.transpose(np.matmul(W3_01,W4_01))
  return np.sum(W_GL_01!=0)

epochs1=100000
epochs4=100000
dim_h=50
dim_z=2

LBDA_L1_n=3
LBDA_L1=.2

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if NOISE:
  x_mat = np.loadtxt('in_data/gauss_train_n.txt')
  n='n_'
  LBDA_L1=LBDA_L1_n
  NP_SEED=1
  OPT_SEED=1
else:
  x_mat = np.loadtxt('in_data/gauss_train.txt')
  n=''
  NP_SEED=8
  OPT_SEED=3

LOG_NAME='gauss_l1_'+n+str(NP_SEED)+'_'+str(OPT_SEED)

#Train with l1 penalty until convergence
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs1, LBDA_L1=LBDA_L1, START_BEST=100, LOG_NAME=LOG_NAME, PRINT=True)
os.system('bash ../copy_params.sh phase0_'+LOG_NAME+'_best '+LOG_NAME+'_best')

#Find thershold with 4 remaining connections
zero_thresh=0.01
while count_conns('phase0_'+LOG_NAME+'_best',zero_thresh)>4:
  zero_thresh+=0.01

#Threshold and train without penalty with fixed zeros until convergennce
path_pen(x_mat=x_mat, y_mat=x_mat, NP_SEED=NP_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase0_'+LOG_NAME+'_best', DIM_H=dim_h, DIM_Z=dim_z, EPOCHS=epochs4, ZERO_THRESH=zero_thresh, LOG_NAME=LOG_NAME, PRINT=True)
os.system('bash ../copy_params.sh phase3_'+LOG_NAME+'_best '+LOG_NAME+'_best')

