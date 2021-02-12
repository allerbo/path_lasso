def load_params(load_str, bs=True):
  import numpy as np
  W1 = np.load('data/W1_'+load_str+'.npy')
  W2 = np.load('data/W2_'+load_str+'.npy')
  W3 = np.load('data/W3_'+load_str+'.npy')
  W4 = np.load('data/W4_'+load_str+'.npy')
  if bs:
    b1 = np.load('data/b1_'+load_str+'.npy')
    b2 = np.load('data/b2_'+load_str+'.npy')
    b3 = np.load('data/b3_'+load_str+'.npy')
    b4 = np.load('data/b4_'+load_str+'.npy')
    return (W1, W2, W3, W4, b1, b2, b3, b4)
  return (W1, W2, W3, W4)

def rec(x,W1,W2,W3,W4,b1=None,b2=None,b3=None,b4=None):
  import numpy as np
  if not b1 is None:
    return np.matmul(np.tanh(np.matmul(np.matmul(np.tanh(np.matmul(x,W1)+b1),W2)+b2,W3)+b3),W4)+b4
  else:
    return np.matmul(np.tanh(np.matmul(np.matmul(np.tanh(np.matmul(x,W1)),W2),W3)),W4)

def z_rec(x,W1,W2,b1=None,b2=None):
  import numpy as np
  if not b1 is None:
    return np.matmul(np.tanh(np.matmul(x,W1)+b1),W2)+b2
  else:
    return np.matmul(np.tanh(np.matmul(x,W1)),W2)

def r2(y,y_hat):
  import numpy as np
  mse = np.mean(np.square(y-y_hat))
  return 1-mse/np.var(y)

def nor(z):
  import numpy as np
  z_nor = (z-np.mean(z,0))/np.std(z,0)
  z_nor[np.isnan(z_nor)]=0
  return z_nor

def conns(W_GL):
  import numpy as np
  return np.sum(W_GL!=0)

def signed_words(W,pc):
  import numpy as np
  pos_words = np.argwhere(W[:,pc]>0).flatten().tolist()
  neg_words = np.argwhere(W[:,pc]<0).flatten().tolist()
  return(pos_words, neg_words)

def int2word(words_idxs, word_list):
  import numpy as np
  words = np.array(word_list)[words_idxs]
  if len(words)==0:
    return '-'
  return ', '.join(words)

def rec_match(rec_mat, true_mat, k=None, labels=None):
  import numpy as np
  n_correct=0
  n_correct_k=0
  n_correct_label=0
  for x_rec in range(rec_mat.shape[0]):
    l2_errs=[]
    for x_true in range(true_mat.shape[0]):
      l2_errs.append(np.mean(np.square(rec_mat[x_rec,:]-true_mat[x_true,:])))
    if x_rec == np.argsort(l2_errs)[0]:
      n_correct+=1
    if not k is None and x_rec in np.argsort(l2_errs)[:k]:
      n_correct_k+=1
    if not labels is None and labels[x_rec] == labels[np.argmin(l2_errs)]:
      n_correct_label+=1
  return n_correct/true_mat.shape[0], n_correct_k/true_mat.shape[0], n_correct_label/true_mat.shape[0]

def best_match(rec_mat, true_mat):
  import numpy as np
  n_correct=0
  sum_l2_err=0
  for f_rec in range(rec_mat.shape[0]):
    best_l2_err=1000000
    for f_true in range(true_mat.shape[0]):
      l2_err = min(best_l2_err,np.mean(np.square(rec_mat[f_rec,:]-true_mat[f_true,:])))
      if l2_err < best_l2_err:
        best_l2_err = l2_err
        best_true=f_true
    if f_rec==best_true:
      sum_l2_err+=best_l2_err
      n_correct+=1
  return (n_correct/true_mat.shape[0], sum_l2_err/n_correct)

def round1(f):
  return '%s' % float('%.2g' % f)
