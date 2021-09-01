import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(1,'..')
from helper_fcts import rec, z_rec, r2, nor, conns, signed_words, rec_match, round1, int2word, load_params
with open('in_data/words4.txt') as f:
  wordlist = f.read().splitlines()

with open('in_data/categories4.txt') as f:
  catlist = f.read().splitlines()


ls=np.loadtxt('in_data/news_lab_val.txt')
x_val=np.loadtxt('in_data/news_val.txt')
colors= ['blue', 'green', 'red', 'cyan']
cs = list(map(lambda e:colors[int(e)],ls))
ALPHA=1
plt.rcParams.update({'legend.fontsize':17})

#blue: windows
#green: hockey
#red: space
#cyan: christian
DIMS=4
FS=40


for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

#Load PL
W1_pl, W2_pl, W3_pl, W4_pl, b1_pl, b2_pl, b3_pl, b4_pl= load_params('phase4_news_pl'+str(DIMS)+'_best')
z_pl=nor(z_rec(x_val,W1_pl,W2_pl,b1_pl,b2_pl))
W_GL_pl = np.matmul(np.square(W1_pl),np.square(W2_pl))+ np.transpose(np.matmul(np.square(W3_pl),np.square(W4_pl)))
W_pl=np.matmul(W1_pl,W2_pl)+np.matmul(W3_pl,W4_pl).T
pos_words_pl0, neg_words_pl0 = signed_words(W_pl,0)
pos_words_pl1, neg_words_pl1 = signed_words(W_pl,1)

#Load L1
W1_l1, W2_l1, W3_l1, W4_l1, b1_l1, b2_l1, b3_l1, b4_l1= load_params('phase4_news_l1'+str(DIMS)+'_best')
z_l1=nor(z_rec(x_val,W1_l1,W2_l1,b1_l1,b2_l1))
W_GL_l1 = np.matmul(np.square(W1_l1),np.square(W2_l1))+ np.transpose(np.matmul(np.square(W3_l1),np.square(W4_l1)))
W_l1=np.matmul(W1_l1,W2_l1)+np.matmul(W3_l1,W4_l1).T
pos_words_l10, neg_words_l10 = signed_words(W_l1,0)
pos_words_l11, neg_words_l11 = signed_words(W_l1,1)

#Load SP
W_sp= np.load('data/W_news_sp'+str(DIMS)+'.npy')
z_sp=nor(np.matmul(x_val,W_sp))
pos_words_sp0, neg_words_sp0 = signed_words(W_sp,0)
pos_words_sp1, neg_words_sp1 = signed_words(W_sp,1)

if DIMS<25:
  for d in range(DIMS):
    pos_words_pl, neg_words_pl = signed_words(W_pl,d)
    pos_words_l1, neg_words_l1 = signed_words(W_l1,d)
    pos_words_sp, neg_words_sp = signed_words(W_sp,d)
  
  print('')
  for alg, W in zip(['Path Lasso', 'Standard Lasso', 'Sparse PCA'], [W_pl, W_l1, W_sp]):
    for d in range(DIMS):
      pos_words, neg_words = signed_words(W,d)
      alg_print='\\multirow{'+str(2*DIMS)+'}{*}{'+alg+'}' if d==0 else ''
      dim_print='\\multirow{2}{*}{'+str(d+1)+ ' of '+str(DIMS)+ '}'
      sign_print_n='Negative'
      words_print_n='\\makecell{' + int2word(neg_words, wordlist) +'}'
      cline_print_n='\\cline{3-4}'
      sign_print_p='Positive'
      words_print_p='\\makecell{' + int2word(pos_words, wordlist) +'}'
      cline_print_p='\\cline{2-4}'
      print(alg_print+' & '+dim_print+' & '+sign_print_n+' & '+words_print_n+'\\\\')
      print(cline_print_n)
      print(' & '+' & '+sign_print_p+' & '+words_print_p+'\\\\')
      print(cline_print_p)
    print('\\hline')
    print('\\hline')
  print('')


np.random.seed(0)
if DIMS==4:
  fig, axs = plt.subplots(3,4,figsize=(32, 27))
  for ax in axs.reshape(-1):
    ax.set_xticks([],[])
    ax.set_yticks([],[])
  for i in range(4):
    for c in range(4):
      color = colors[c]
      cat = catlist[c]
      z_sub = z_pl[np.argwhere(ls==c).squeeze()]
      axs[0,i].scatter(z_sub[:,i],np.random.uniform(0,1,z_sub.shape[0]),c=color,label=cat,alpha=ALPHA)

    for c in range(4):
      color = colors[c]
      cat = catlist[c]
      z_sub = z_l1[np.argwhere(ls==c).squeeze()]
      axs[1,i].scatter(z_sub[:,i],np.random.uniform(0,1,z_sub.shape[0]),c=color,label=cat,alpha=ALPHA)
    
    for c in range(4):
      color = colors[c]
      cat = catlist[c]
      z_sub = z_sp[np.argwhere(ls==c).squeeze()]
      axs[2,i].scatter(z_sub[:,i],np.random.uniform(0,1,z_sub.shape[0]),c=color,label=cat,alpha=ALPHA)
    
    axs[0,i].legend()
    axs[1,i].legend()
    axs[2,i].legend()
    axs[0,i].set_title('\nPath Lasso, Dim '+str(i+1),fontsize=FS)
    axs[1,i].set_title('\nStandard Lasso, Dim '+str(i+1),fontsize=FS)
    axs[2,i].set_title('\nSparse PCA, Dim '+str(i+1),fontsize=FS)
if DIMS==2:
  fig, axs = plt.subplots(1,3,figsize=(23, 9))
  for ax in axs:
    ax.set_xticks([],[])
    ax.set_yticks([],[])
  axs=axs.ravel()
  for c in range(4):
    color = colors[c]
    cat = catlist[c]
    z_sub = z_pl[np.argwhere(ls==c).squeeze()]
    axs[0].scatter(z_sub[:,0],z_sub[:,1],c=color,label=cat,alpha=ALPHA)

  for c in range(4):
    color = colors[c]
    cat = catlist[c]
    z_sub = z_l1[np.argwhere(ls==c).squeeze()]
    axs[1].scatter(z_sub[:,0],z_sub[:,1],c=color,label=cat,alpha=ALPHA)
  
  for c in range(4):
    color = colors[c]
    cat = catlist[c]
    z_sub = z_sp[np.argwhere(ls==c).squeeze()]
    axs[2].scatter(z_sub[:,0],z_sub[:,1],c=color,label=cat,alpha=ALPHA)
  
  
  axs[0].legend()
  axs[1].legend()
  axs[2].legend()
  axs[0].set_title('\nPath Lasso',fontsize=FS)
  axs[1].set_title('\nStandard Lasso',fontsize=FS)
  axs[2].set_title('\nSparse PCA',fontsize=FS)
  
if DIMS<25:
  fig.tight_layout(pad=5)
  plt.savefig('figures/news'+str(DIMS)+'.pdf')

rec_pl=rec(x_val,W1_pl,W2_pl,W3_pl,W4_pl,b1_pl,b2_pl,b3_pl,b4_pl)
rm_pl, _, rm_cl_pl = rec_match(rec_pl, x_val, None, ls)
rec_l1=rec(x_val,W1_l1,W2_l1,W3_l1,W4_l1,b1_l1,b2_l1,b3_l1,b4_l1)
rm_l1, _, rm_cl_l1 = rec_match(rec_l1, x_val, None, ls)
rec_sp=np.matmul(np.matmul(x_val,W_sp),W_sp.T)
rm_sp, _, rm_cl_sp = rec_match(rec_sp, x_val, None, ls)

print('Dimensions & Algorithm & Connections & $R^2$ & Reconstruction Match \\\\')
print('\\hline')
print(str(DIMS)+ ' & Path Lasso & ' +str(conns(W_GL_pl))+' & '+ round1(r2(x_val, rec_pl))+ ' & ' + round1(rm_pl)+ ' & ' + round1(rm_cl_pl) + '\\\\')
print(str(DIMS)+ ' & Standard Lasso & ' +str(conns(W_GL_l1))+' & '+ round1(r2(x_val, rec_l1))+ ' & ' + round1(rm_l1)+ ' & ' + round1(rm_cl_l1) + '\\\\')
print(str(DIMS)+ ' & Sparse PCA & ' +str(conns(W_sp))+' & '+ round1(r2(x_val, rec_sp))+ ' & ' + round1(rm_sp)+' & ' + round1(rm_cl_sp)+ '\\\\')
