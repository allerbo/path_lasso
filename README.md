This is the code used in the article **Non-linear, Sparse Dimensionality Reduction via Path Lasso Penalized Autoencoders**, available at https://arxiv.org/abs/2102.10873 and https://www.jmlr.org/papers/volume22/21-0203/21-0203.pdf.

Code in *synth_gauss* was run using Python 3.7.4 and TensorFlow 2.1.0, while code in *news* was run using Python 3.8.7 and TensorFlow 2.4.1. This is only important to get exact reconstruction of the images, and there might still be descripancies due to different hardware use.

The code in omf.py and cdnmf_fast.pyx is heavily based on the non-negative matrix factorization code in python's scikit-learn module.

## Compiling cdnmf_fast:
```
$ cythonize -i cdnmf_fast.pyx
```
## Figure 2:
```
$ cd synth_gauss
$ python make_synth_gauss.py                   #Create synthetic data
$ python gauss_pl.py NOISE=False               #Path Lasso without noise
$ python gauss_pl.py NOISE=True                #Path Lasso with noise
$ python gauss_l1.py NOISE=False               #Standard Lasso without noise
$ python gauss_l1.py NOISE=True                #Standard Lasso with noise
$ python gauss_ae.py NOISE=False               #Autoencoder without noise
$ python gauss_ae.py NOISE=True                #Autoencoder with noise
$ python gauss_spae.py NOISE=False             #Sparse Autoencoder without noise
$ python gauss_spae.py NOISE=True              #Sparse Autoencoder with noise
$ python gauss_sp_pca.py NOISE=False SP=False  #PCA without noise
$ python gauss_sp_pca.py NOISE=True SP=False   #PCA with noise
$ python gauss_sp_pca.py NOISE=False SP=True   #Sparse PCA without noise
$ python gauss_sp_pca.py NOISE=True SP=True    #Sparse PCA wit noise
$ python plot_gauss.py                         #Plot figure
```
## Figure 3 and Tables 2-4:
```
$ cd news
$ python extract_news.py               #Extract data
$ python news_pl2.py                   #Path Lasso, 2 dimensions
$ python news_pl4.py                   #Path Lasso, 4 dimensions
$ python news_pl25.py                  #Path Lasso, 25 dimensions
$ python news_l12.py                   #Standard Lasso, 2 dimensions
$ python news_l14.py                   #Standard Lasso, 4 dimensions
$ python news_l125.py                  #Standard Lasso, 25 dimensions
$ python news_sp2.py                   #Sparse PCA, 2 dimensions
$ python news_sp4.py                   #Sparse PCA, 4 dimensions
$ python news_sp25.py                  #Sparse PCA, 25 dimensions
$ python tables_plots_news.py DIMS=2   #Generate tables and plots
$ python tables_plots_news.py DIMS=4   #Generate tables and plots
$ python tables_plots_news.py DIMS=25  #Generate table
```

