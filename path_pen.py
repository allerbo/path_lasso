def path_pen(x_mat, y_mat, NP_SEED, TF_SEED, BATCH_DIV, LOAD_APDX, DIM_H, DIM_Z, EPOCHS, LOG_NAME='', LBDA_KL=None, LBDA_L2=None, LBDA_L1=None, LBDA_EXCL=None, LBDA_PATH=None, LBDA_PATH_PROX=None, STEP_SIZE=None, ZERO_THRESH=None, PRINT=True, PAR_SPLITS=None, USE_BIAS=True, START_BEST=5):

  import numpy as np
  import tensorflow as tf
  import omf
  import os,sys
  from collections import Counter
  import itertools as it
  import threading, time, datetime

  def load1(load_str):
    if os.path.exists(load_str+'.npy'):
      return np.load(load_str+'.npy')
    return np.loadtxt(load_str+'.txt')


  class model:

    ###Init
    def __init__(self, DIM_H, DIM_Y, DIM_Z, USE_BIAS, LBDA_KL, LBDA_L2, LBDA_L1, LBDA_EXCL, LBDA_PATH, LBDA_PATH_PROX, STEP_SIZE):
      #tf.keras.backend.set_floatx('float64')
      self.h1=tf.keras.layers.Dense(DIM_H, activation='tanh', use_bias=USE_BIAS)
      self.z=tf.keras.layers.Dense(DIM_Z, activation='linear', use_bias=USE_BIAS)
      self.h3=tf.keras.layers.Dense(DIM_H, activation='tanh', use_bias=USE_BIAS)
      self.y=tf.keras.layers.Dense(DIM_Y, activation='linear', use_bias=USE_BIAS)
      self.train_op = tf.keras.optimizers.Adam()
      
      self.KL_RHO=1/DIM_Z
      self.LBDA_KL=LBDA_KL
      self.LBDA_L2=LBDA_L2
      self.LBDA_L1=LBDA_L1
      self.LBDA_EXCL=LBDA_EXCL
      self.LBDA_PATH=LBDA_PATH
      self.GAMMA=2.
      self.LBDA_PATH_PROX=LBDA_PATH_PROX
      self.GAMMA_PROX=2.
      self.STEP_SIZE=STEP_SIZE
      self.USE_BIAS=USE_BIAS

      self.r2_count=0
      self.old_r2=-10000

    def run(self,x):
      return self.y(self.h3(self.z(self.h1(x))))
     
    def initialize(self, x):
      self.run(x)
      self.W1 = self.h1.variables[0]
      self.W2 = self.z.variables[0]
      self.W3 = self.h3.variables[0]
      self.W4 = self.y.variables[0]
      if self.USE_BIAS:
        self.b1 = self.h1.variables[1]
        self.b2 = self.z.variables[1]
        self.b3 = self.h3.variables[1]
        self.b4 = self.y.variables[1]

    def safe_div(self,a,b):
      MIN_FLOAT = 1e-30
      return np.divide(a,np.maximum(MIN_FLOAT,b))

    def safe_log(self,a):
      MIN_FLOAT = 1e-30
      return tf.math.log(tf.maximum(a,MIN_FLOAT))

    ###Setters and getters
    def set_Ws(self,Ws):
      tf.keras.backend.set_value(self.W1,Ws[0])
      tf.keras.backend.set_value(self.W2,Ws[1])
      tf.keras.backend.set_value(self.W3,Ws[2])
      tf.keras.backend.set_value(self.W4,Ws[3])
    
    def set_bs(self,bs):
      tf.keras.backend.set_value(self.b1,bs[0])
      tf.keras.backend.set_value(self.b2,bs[1])
      tf.keras.backend.set_value(self.b3,bs[2])
      tf.keras.backend.set_value(self.b4,bs[3])

    def get_Ws(self):
      Ws = [self.W1.numpy(), self.W2.numpy(), self.W3.numpy(), self.W4.numpy()]
      return Ws

    def get_bs(self):
      bs = [self.b1.numpy(), self.b2.numpy(), self.b3.numpy(), self.b4.numpy()]
      return bs


    def get_W_GL(self, TF=False):
      W_GL2=tf.matmul(tf.square(self.W1),tf.square(self.W2))+tf.transpose(tf.matmul(tf.square(self.W3),tf.square(self.W4)))
      if TF:
        return tf.sqrt(W_GL2)
      return tf.sqrt(W_GL2).numpy()

    ###Losses
    def r_squared(self,x,y):
      ss_tot = tf.cast(tf.reduce_mean(tf.square(y-tf.reduce_mean(y))),dtype=tf.float32)
      return 1-self.mse_loss(x,y)/ss_tot

    def mse_loss(self,x,y):
      y_hat=self.run(x)
      return tf.reduce_mean(tf.square(y_hat-y))

    def kl_loss(self, x):
      z_01=tf.abs(tf.tanh(self.z(self.h1(x))))
      rho_hat = tf.reduce_mean(z_01, axis=0)
      kl_div =self.KL_RHO * self.safe_log(self.KL_RHO) - self.KL_RHO * self.safe_log(rho_hat) + (1 - self.KL_RHO) * self.safe_log(1 - self.KL_RHO) - (1 - self.KL_RHO) * self.safe_log(1 - rho_hat)
      return self.LBDA_KL*kl_div

    def rho_hat(self, x):
      z_01=tf.abs(tf.tanh(self.z(self.h1(x))))
      rho_hat = tf.reduce_mean(z_01, axis=0)
      return rho_hat

    def l2_loss(self):
      l2_loss = tf.reduce_mean(tf.square(self.W1)) + tf.reduce_mean(tf.square(self.W2)) + tf.reduce_mean(tf.square(self.W3)) + tf.reduce_mean(tf.square(self.W4))
      return self.LBDA_L2*l2_loss

    def l1_loss(self):
      l1_loss = tf.reduce_mean(tf.abs(self.W1)) + tf.reduce_mean(tf.abs(self.W2)) +  tf.reduce_mean(tf.abs(self.W3)) + tf.reduce_mean(tf.abs(self.W4))
      return self.LBDA_L1*l1_loss

    def excl_loss(self):
      W_GL=self.get_W_GL(TF=True)
      excl_loss = tf.reduce_mean(tf.square(tf.reduce_sum(W_GL,1)))
      return self.LBDA_EXCL*excl_loss

    def start_path(self, LOAD_APDX):
      W_GL = self.get_W_GL()
      if LOAD_APDX is None:
        self.lbda_mat_path = np.multiply(self.LBDA_PATH,np.ones(W_GL.shape))
      else:
        self.lbda_mat_path = self.safe_div(self.LBDA_PATH,np.power(W_GL,self.GAMMA))
     
    def path_loss(self):
      W_GL=self.get_W_GL(TF=True)
      path_loss = tf.reduce_mean(tf.multiply(self.lbda_mat_path,W_GL))
      return path_loss

    def start_fixed_zeros(self, thresh):
      W1, W2, W3, W4 = self.get_Ws()
      self.W1_01 = 1*(np.abs(W1)>thresh)
      self.W2_01 = 1*(np.abs(W2)>thresh)
      self.W3_01 = 1*(np.abs(W3)>thresh)
      self.W4_01 = 1*(np.abs(W4)>thresh)
      self.train_op = tf.keras.optimizers.Adam()
      
    def set_fixed_zeros(self):
      W1, W2, W3, W4 = self.get_Ws()
      W_zeros = [np.multiply(self.W1_01,W1), np.multiply(self.W2_01,W2), np.multiply(self.W3_01,W3), np.multiply(self.W4_01,W4)]
      self.set_Ws(W_zeros)

    ###Training
    def get_grad(self,x,y):
      with tf.GradientTape() as tape:
        L = self.mse_loss(x,y)
        if not self.LBDA_KL is None:
          L += self.kl_loss(x)
        if not self.LBDA_L2 is None:
          L += self.l2_loss()
        if not self.LBDA_L1 is None:
          L += self.l1_loss()
        if not self.LBDA_EXCL is None:
          L += self.excl_loss()
        if not self.LBDA_PATH is None:
          L += self.path_loss()
        Wbs = [self.W1,self.W2,self.W3,self.W4]
        if self.USE_BIAS:
          Wbs += [self.b1,self.b2,self.b3,self.b4]
        g = tape.gradient(L, Wbs)
      return g
     
    def train(self,x,y):
      g = self.get_grad(x,y)
      Wbs = [self.W1,self.W2,self.W3,self.W4]
      if self.USE_BIAS:
        Wbs += [self.b1,self.b2,self.b3,self.b4]
      self.train_op.apply_gradients(zip(g, Wbs))
     
    ###Save and print
    def save_par(self, x_train, x_test, apdx=''):
      W1, W2, W3, W4 = self.get_Ws()
      np.save('data/W1_'+LOG_NAME+apdx,W1)
      np.save('data/W2_'+LOG_NAME+apdx,W2)
      np.save('data/W3_'+LOG_NAME+apdx,W3)
      np.save('data/W4_'+LOG_NAME+apdx,W4)
      if self.USE_BIAS:
        b1, b2, b3, b4 = self.get_bs()
        np.save('data/b1_'+LOG_NAME+apdx,b1)
        np.save('data/b2_'+LOG_NAME+apdx,b2)
        np.save('data/b3_'+LOG_NAME+apdx,b3)
        np.save('data/b4_'+LOG_NAME+apdx,b4)

    def r2_best(self, x_train, x_test, y_test):
      r2 = self.r_squared(x_test,y_test)
      if r2>self.old_r2:
        self.save_par(x_train, x_test, '_best')
        self.old_r2 = self.r_squared(x_test,y_test)
        self.r2_count = 0
      else:
        self.r2_count += 1
      return self.r2_count

    def print_str(self,x,y,epoch):
      print_str =  'Epoch: '+str(epoch)+'\n'
      print_str += 'Error r^2: '+str(self.r_squared(x,y).numpy())+'\n'
      if not self.LBDA_KL is None:
        print_str += 'Error KL ('+str(self.LBDA_KL)+'): '+str(self.kl_loss(x).numpy())+'\n'
        print_str += 'Rho_hat ('+str(self.KL_RHO)+'): '+str(self.rho_hat(x).numpy())+'\n'
      if not self.LBDA_L2 is None:
        print_str += 'Error L2 ('+str(self.LBDA_L2)+'): '+str(self.l2_loss().numpy())+'\n'
      if not self.LBDA_L1 is None:
        print_str += 'Error L1 ('+str(self.LBDA_L1)+'): '+str(self.l1_loss().numpy())+'\n'
      if not self.LBDA_EXCL is None:
        print_str += 'Error excl ('+str(self.LBDA_EXCL)+'): '+str(self.excl_loss().numpy())+'\n'
      if not self.LBDA_PATH is None:
        print_str += 'Error path ('+str(self.LBDA_PATH)+'): '+str(self.path_loss().numpy())+'\n'
      if not self.LBDA_PATH_PROX is None:
        print_str += 'Path prox: '+str(self.LBDA_PATH_PROX)+'. Step size: '+str(self.STEP_SIZE)+'.\n'
      W1, W2, W3, W4 = self.get_Ws()
      print_str += 'W1: '+str(np.sum(np.abs(W1)>0.001))+' '+str(np.sum(np.abs(W1)>0))+'\n'
      print_str += 'W2: '+str(np.sum(np.abs(W2)>0.001))+' '+str(np.sum(np.abs(W2)>0))+'\n'
      print_str += 'W3: '+str(np.sum(np.abs(W3)>0.001))+' '+str(np.sum(np.abs(W3)>0))+'\n'
      print_str += 'W4: '+str(np.sum(np.abs(W4)>0.001))+' '+str(np.sum(np.abs(W4)>0))+'\n'
      W_GL = self.get_W_GL()
      print_str += 'Used X: '+ str(np.sum(np.sum(W_GL>0.001,1)!=0))+ ' ' + str(np.sum(np.sum(W_GL!=0,1)!=0))+'\n'
      print_str += 'Used Z: '+ str(np.sum(np.sum(W_GL>0.001,0)!=0))+ ' ' + str(np.sum(np.sum(W_GL!=0,0)!=0))+'\n'
      print_str += 'W_GL: '+str(np.sum(W_GL>0.001))+' '+str(np.sum(W_GL>0))+'\n'
      print_str += str(W_GL[:10,:6])+'\n'
      print_str += str(np.sum(W_GL,0))+'\n'
      return print_str
 

    ###Proximal
    def start_path_prox(self, par_splits):
      W_GL = self.get_W_GL()
      self.lbda_mat_path_prox = self.safe_div(self.LBDA_PATH_PROX,np.power(W_GL,self.GAMMA_PROX))
      self.train_op = tf.keras.optimizers.SGD(self.STEP_SIZE)
      if par_splits is None:
        self.n_threads=0
      else:
        self.start_threads(par_splits)

    def start_threads(self,par_splits):
      self.b_stop_threads = False
      W1, W2 = self.get_Ws()[:2]
      idxs1=self.get_thread_idxs(W1.shape[0],par_splits[0])
      idxs2=self.get_thread_idxs(W1.shape[1],par_splits[1])
      idxs3=self.get_thread_idxs(W2.shape[1],par_splits[2])
      self.index_combs = list(it.product(idxs1,idxs2,idxs3))
      self.n_threads = len(self.index_combs)
      self.events_start = [threading.Event() for i in range(self.n_threads)]
      self.events_done =  [threading.Event() for i in range(self.n_threads)]
      
      self.lock_SP=threading.Lock()
      self.lock1=threading.Lock()
      self.lock2=threading.Lock()
      self.lock3=threading.Lock()
      self.lock4=threading.Lock()
      for thread_idx in range(self.n_threads):
        threading.Thread(target=self.split_and_prox, args=(thread_idx,1)).start()

    def stop_threads(self):
      if self.n_threads>0:
        self.b_stop_threads = True
        for thread_idx in range(self.n_threads):
          self.events_start[thread_idx].set()

    def get_thread_idxs(self, shape, n_split):
      per_thread = int(shape//n_split)
      if shape%n_split > 0:
        per_thread += 1 
      starts = list(range(0,shape,per_thread))
      stops = list(map(lambda start: start+per_thread, starts))
      stops[-1] = shape
      idxs = []
      for start, stop in zip(starts, stops):
        idxs.append(list(range(start, stop)))
      return(idxs)

    def split_and_prox(self, my_id, dummy):
      while True: #Stay on standby until self.stop is called
        self.events_start[my_id].wait()
        if self.b_stop_threads:
          return
        idxs1, idxs2, idxs3 = self.index_combs[my_id]

        W_GL = self.get_W_GL()
        W_GL_th = W_GL[np.ix_(idxs1,idxs3)]
        lbda_mat_path_prox_th = self.lbda_mat_path_prox[np.ix_(idxs1,idxs3)]

        W1, W2, W3, W4 = self.get_Ws()
        W1_th = W1[np.ix_(idxs1,idxs2)]
        W2_th = W2[np.ix_(idxs2,idxs3)]
        W1_pen_abs, W2_pen_abs, W12_SP_pen = self.do_prox_path(W1_th, W2_th, W_GL_th, lbda_mat_path_prox_th, False,my_id)
        self.lock_SP.acquire()
        self.W12_SP[np.ix_(idxs1,idxs3)] += W12_SP_pen
        self.lock_SP.release()
        self.lock1.acquire()
        self.Ws_big[0][np.ix_(idxs1,idxs2)] = np.maximum(self.Ws_big[0][np.ix_(idxs1,idxs2)], W1_pen_abs)
        self.lock1.release()
        self.lock2.acquire()
        self.Ws_big[1][np.ix_(idxs2,idxs3)] = np.maximum(self.Ws_big[1][np.ix_(idxs2,idxs3)], W2_pen_abs)
        self.lock2.release()
        W3_th = W3[np.ix_(idxs3,idxs2)]
        W4_th = W4[np.ix_(idxs2,idxs1)]
        W3_pen_abs, W4_pen_abs, W34_SP_pen = self.do_prox_path(W3_th, W4_th, W_GL_th, lbda_mat_path_prox_th, True, my_id)
        self.lock_SP.acquire()
        self.W34_SP[np.ix_(idxs3,idxs1)] += W34_SP_pen
        self.lock_SP.release()
        self.lock3.acquire()
        self.Ws_big[2][np.ix_(idxs3,idxs2)] = np.maximum(self.Ws_big[2][np.ix_(idxs3,idxs2)], W3_pen_abs)
        self.lock3.release()
        self.lock4.acquire()
        self.Ws_big[3][np.ix_(idxs2,idxs1)] = np.maximum(self.Ws_big[3][np.ix_(idxs2,idxs1)], W4_pen_abs)
        self.lock4.release()
        self.events_start[my_id].clear()
        self.events_done[my_id].set()
   

    def prox_path(self, par_splits):
      Ws = self.get_Ws()
      if not par_splits is None:
        self.Ws_big=list(map(lambda W: np.zeros(W.shape),Ws))
        self.W12_SP = np.zeros((Ws[0].shape[0], Ws[1].shape[1]))
        self.W34_SP = np.zeros((Ws[2].shape[0], Ws[3].shape[1]))
        #Activate threads
        t_start_th = datetime.datetime.now()
        for thread_idx in range(self.n_threads):
          self.events_done[thread_idx].clear()
          self.events_start[thread_idx].set()

        #Wait until all threads finished
        for event in self.events_done:
          event.wait()
        t_end_th = datetime.datetime.now()

        W1_pen_abs=self.Ws_big[0]#/par_splits[2]
        W2_pen_abs=self.Ws_big[1]#/par_splits[0]
        W12_SP_pen=self.W12_SP
        W3_pen_abs=self.Ws_big[2]#/par_splits[2]
        W4_pen_abs=self.Ws_big[3]#/par_splits[0]
        W34_SP_pen=self.W34_SP
      else:
        W_GL = self.get_W_GL()
        W1, W2 = Ws[:2]
        W1_pen_abs, W2_pen_abs, W12_SP_pen = self.do_prox_path(W1, W2, W_GL, self.lbda_mat_path_prox, False)
        W3, W4 = Ws[2:]
        W3_pen_abs, W4_pen_abs, W34_SP_pen = self.do_prox_path(W3, W4, W_GL, self.lbda_mat_path_prox, True)
      #end if
      W1_pen_abs_t, W2_pen_abs_t = self.do_thresh(W1_pen_abs, W2_pen_abs, W12_SP_pen)
      W_pens = (np.multiply(np.sign(Ws[0]), W1_pen_abs_t), np.multiply(np.sign(Ws[1]), W2_pen_abs_t))
      W3_pen_abs_t, W4_pen_abs_t = self.do_thresh(W3_pen_abs, W4_pen_abs, W34_SP_pen)
      W_pens += (np.multiply(np.sign(Ws[2]), W3_pen_abs_t), np.multiply(np.sign(Ws[3]), W4_pen_abs_t))
      if self.n_threads>0:
        print1(str(t_end_th-t_start_th))
      self.set_Ws(W_pens)

    def do_prox_path(self, W1, W2, W_GL, lbda_mat_path_prox, dec=False,my_id=0):
      #Path: penalize:
      #W_GL = sqrt(W1.^2*W2.^2)
      #lbda_mat_path_prox = LBDA/init_weights_norm.^GAMMA
      #init_weights = sqrt(W1_init.^2*W2_init.^2)
      #W_SP = |W1|*|W2|
      #W_penalty = max(0,1-alpha*lbda_mat_path_prox/W_GL)
      #W_SP_penalized=W_SP.*W_penalty
      #W1_penalized_abs, W2_penalized_abs = omf(W_penalized, |W1|, |W2|)
      #W1_penalized = sgn(W1).*W1_penalized_abs
      #W2_penalized = sgn(W2).*W2_penalized_abs

      #W1, W2 = self.get_Ws()[:2]
      #W_GL = self.get_W_GL()
      W_SP = np.matmul(np.abs(W1),np.abs(W2)) #Sum paths
      W_penalty = np.maximum(0,1-self.STEP_SIZE*self.safe_div(lbda_mat_path_prox,W_GL))
      if dec:
        W_SP_penalized = np.multiply(W_SP, W_penalty.T)
      else:
        W_SP_penalized = np.multiply(W_SP, W_penalty)

      (W1_penalized_abs, W2_penalized_abs, n_iter) = omf.factorize(W_SP_penalized, np.abs(W1), np.abs(W2), tol=1e-4)
      return (W1_penalized_abs, W2_penalized_abs, W_SP_penalized)

    def do_thresh(self, W1_penalized_abs, W2_penalized_abs, W_SP_penalized):
      threshs = 1./np.flip(np.power(np.sqrt(10),range(21)))
      vals = np.zeros(len(threshs))
      for t in range(len(threshs)):
        thresh = threshs[t]
        val = np.sum(1.*(1*(W_SP_penalized>0) != 1.*(np.matmul(1.*(W1_penalized_abs>thresh),1.*(W2_penalized_abs>thresh))>0)))
        if val == 0:
          opt_thresh=thresh
          break
        vals[t]=val
      opt_thresh = threshs[np.argmin(vals)]
      #opt_thresh=1e-100
      W1_penalized_abs[W1_penalized_abs<=opt_thresh]=0.
      W2_penalized_abs[W2_penalized_abs<=opt_thresh]=0.
      return (W1_penalized_abs, W2_penalized_abs)


  #Print to logfile if on cluster, else to just print
  def print1(print_str):
    if 'SLURM_SUBMIT_DIR' in os.environ:
       with open(os.environ['SLURM_SUBMIT_DIR']+'/logs/'+LOG_NAME+'.txt', 'a+') as f:
         f.write(str(print_str) + '\n')
    else:
      print(print_str)
  
  def xy_batch(x_train, y_train, batch, batch_size):
    start = batch * batch_size
    stop = (batch + 1) * batch_size
    x_batch = x_train[start:stop]
    y_batch = y_train[start:stop]
    return (x_batch, y_batch)

  if not NP_SEED is None:
    np.random.seed(NP_SEED)
  else:
    np.random.seed()
  
  if not TF_SEED is None:
    tf.random.set_seed(TF_SEED)

  np.set_printoptions(suppress=True)
  DIM_X = x_mat.shape[1] #Number of x-dimensions
  DIM_Y = y_mat.shape[1] #Number of y-dimensions

  N_SAMPLES = x_mat.shape[0]
  assert N_SAMPLES == y_mat.shape[0]
  
  #Split in test and training data (90%  train and 10% test)
  n_train = int(round(0.9 * N_SAMPLES))
  p = np.random.permutation(N_SAMPLES)
  x_train, x_test = x_mat[p][:n_train,:], x_mat[p][n_train:,:] 
  y_train, y_test = y_mat[p][:n_train,:], y_mat[p][n_train:,:] 
  BATCH_SIZE = int(round(1.*x_train.shape[0]/BATCH_DIV)) #Number of samples used in each gradient descent update
  
  #Set up Tensorflow
  my_model = model(DIM_H, DIM_Y, DIM_Z, USE_BIAS, LBDA_KL, LBDA_L2, LBDA_L1, LBDA_EXCL, LBDA_PATH, LBDA_PATH_PROX, STEP_SIZE)
  my_model.initialize(x_test) #initialize
  if not LOAD_APDX is None:
    W1 = load1('data/W1_'+LOAD_APDX)
    W2 = load1('data/W2_'+LOAD_APDX)
    W3 = load1('data/W3_'+LOAD_APDX)
    W4 = load1('data/W4_'+LOAD_APDX)
    Ws = [W1,W2,W3,W4]
    if USE_BIAS:
      b1 = load1('data/b1_'+LOAD_APDX)
      b2 = load1('data/b2_'+LOAD_APDX)
      b3 = load1('data/b3_'+LOAD_APDX)
      b4 = load1('data/b4_'+LOAD_APDX)
      bs = [b1,b2,b3,b4]

    my_model.set_Ws(Ws)
    if USE_BIAS:
      my_model.set_bs(bs)
  
  if PRINT:
    if 'SLURM_JOB_ID' in os.environ:
      print1(os.environ['SLURM_JOB_ID'])
    print1(LOAD_APDX)
  if not LBDA_PATH is None:
    my_model.start_path(LOAD_APDX)
  if not LBDA_PATH_PROX is None:
    my_model.start_path_prox(PAR_SPLITS)
  if not ZERO_THRESH is None:
    my_model.start_fixed_zeros(ZERO_THRESH)
  for epoch in range(EPOCHS):
    if (epoch % 100 == 0) or (not ZERO_THRESH is None and epoch % 100 == 0):
      my_model.save_par(x_train, x_test)
      if epoch > START_BEST:
        r2_count = my_model.r2_best(x_train, x_test, y_test)
        if PRINT:
          print1(str(r2_count))
        if r2_count>10:
          break
      if PRINT:
        print1(my_model.print_str(x_test,y_test,epoch))

    #run for mini-batches
    for batch in range(-(-x_train.shape[0] // BATCH_SIZE)): #two minus signs to get ceiling division (instead of floor)
      my_model.train(*xy_batch(x_train, y_train, batch, BATCH_SIZE))
      if not LBDA_PATH_PROX is None:
        my_model.prox_path(PAR_SPLITS)
      if not ZERO_THRESH is None:
        my_model.set_fixed_zeros()

  if not PAR_SPLITS is None:
    my_model.stop_threads()
  my_model.save_par(x_train, x_test)
  Ws = my_model.get_Ws()
  W_GL = my_model.get_W_GL()
  return (*Ws, W_GL)
