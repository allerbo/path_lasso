import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(1,'..')
from helper_fcts import load_params, nor, z_rec
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)

def mscatter(xy,ax, m, c, alpha, title):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(xy[:,0],xy[:,1],c=c,alpha=alpha)
    if (m is not None) and (len(m)==len(xy)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    if not title is None:
      ax.set_title(title,fontsize=40)
    ax.axis('equal')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    return sc

ALPHA=1

np.random.seed(0)
x_val=np.loadtxt('in_data/gauss_val.txt')
x_val_n=np.loadtxt('in_data/gauss_val_n.txt')
ls=np.loadtxt('in_data/gauss_lab_val.txt')
ls_train=np.loadtxt('in_data/gauss_lab_train.txt')


cs=[]
ms=[]
for l in ls:
  if l > 7:
    cs.append('C'+str(int(15-l)))
    ms.append('o')
  else:
    cs.append('C'+str(int(l)))
    ms.append('x')



W1_ae, W2_ae, W3_ae, W4_ae, b1_ae, b2_ae, b3_ae, b4_ae= load_params('phase0_gauss_ae_4_1_best')
z_ae=z_rec(x_val, W1_ae, W2_ae, b1_ae, b2_ae)

W1_spae, W2_spae, W3_spae, W4_spae, b1_spae, b2_spae, b3_spae, b4_spae= load_params('phase0_gauss_spae_2_1_best')
z_spae=z_rec(x_val, W1_spae, W2_spae, b1_spae, b2_spae)

W1_pl, W2_pl, W3_pl, W4_pl, b1_pl, b2_pl, b3_pl, b4_pl= load_params('phase3_gauss_pl_6_1_best')
z_pl=z_rec(x_val, W1_pl, W2_pl, b1_pl, b2_pl)

W1_l1, W2_l1, W3_l1, W4_l1, b1_l1, b2_l1, b3_l1, b4_l1= load_params('phase3_gauss_l1_8_3_best')
z_l1=z_rec(x_val, W1_l1, W2_l1, b1_l1, b2_l1)


W_sp= np.load('data/W_gauss_sp.npy')
z_sp=np.matmul(x_val,W_sp)

W_pca= np.load('data/W_gauss_pca.npy')
z_pca=np.matmul(x_val,W_pca)

W1_aen, W2_aen, W3_aen, W4_aen, b1_aen, b2_aen, b3_aen, b4_aen= load_params('phase0_gauss_ae_n_5_2_best')
z_aen=z_rec(x_val_n, W1_aen, W2_aen, b1_aen, b2_aen)

W1_spaen, W2_spaen, W3_spaen, W4_spaen, b1_spaen, b2_spaen, b3_spaen, b4_spaen= load_params('phase0_gauss_spae_n_4_2_best')
z_spaen=z_rec(x_val_n, W1_spaen, W2_spaen, b1_spaen, b2_spaen)

W1_pln, W2_pln, W3_pln, W4_pln, b1_pln, b2_pln, b3_pln, b4_pln= load_params('phase3_gauss_pl_n_2_3_best')
z_pln=z_rec(x_val_n, W1_pln, W2_pln, b1_pln, b2_pln)

W1_l1n, W2_l1n, W3_l1n, W4_l1n, b1_l1n, b2_l1n, b3_l1n, b4_l1n= load_params('phase3_gauss_l1_n_1_1_best')
z_l1n=z_rec(x_val_n, W1_l1n, W2_l1n, b1_l1n, b2_l1n)


W_spn= np.load('data/W_gauss_sp_n.npy')
z_spn=np.matmul(x_val_n,W_spn)

W_pcan= np.load('data/W_gauss_pca_n.npy')
z_pcan=np.matmul(x_val_n,W_pcan)


fig,axs=plt.subplots(3,4,figsize=(30,27))
mscatter(z_pl,axs[0,0],m=ms,c=cs,alpha=ALPHA, title='Path Lasso')
mscatter(z_l1,axs[0,1],m=ms,c=cs,alpha=ALPHA, title='Standard Lasso')
mscatter(z_pln,axs[0,2],m=ms,c=cs,alpha=ALPHA, title='Path Lasso')
mscatter(z_l1n,axs[0,3],m=ms,c=cs,alpha=ALPHA, title='Standard Lasso')
mscatter(z_ae,axs[1,0],m=ms,c=cs,alpha=ALPHA, title='Autoencoder')
mscatter(z_spae,axs[1,1],m=ms,c=cs,alpha=ALPHA, title='Sparse Autoencoder')
mscatter(z_aen,axs[1,2],m=ms,c=cs,alpha=ALPHA, title='Autoencoder')
mscatter(z_spaen,axs[1,3],m=ms,c=cs,alpha=ALPHA, title='Sparse Autoencoder')
mscatter(z_pca,axs[2,0],m=ms,c=cs,alpha=ALPHA, title='PCA')
mscatter(z_sp,axs[2,1],m=ms,c=cs,alpha=ALPHA, title='Sparse PCA')
mscatter(z_pcan,axs[2,2],m=ms,c=cs,alpha=ALPHA, title='PCA')
mscatter(z_spn,axs[2,3],m=ms,c=cs,alpha=ALPHA, title='Sparse PCA')
fig.suptitle('Without Added Noise                                    With Added Noise  ',fontsize=50)
l1 = Line2D([1082,1082], [0, 1900],color='k', linewidth=2)
fig.lines.extend([l1])
fig.tight_layout(pad=5)
plt.savefig('figures/gauss.pdf')
