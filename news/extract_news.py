import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

categories = ['soc.religion.christian', 'sci.space', 'comp.windows.x', 'rec.sport.hockey']
newsgroups_train = fetch_20newsgroups(categories=categories,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories,remove=('headers', 'footers', 'quotes'))
vectorizer_tfidf = TfidfVectorizer(stop_words='english')
vectors_tfidf = vectorizer_tfidf.fit_transform(newsgroups_train.data)
feature_names = np.asarray(vectorizer_tfidf.get_feature_names())

top_tfidf=np.flip(np.argsort(np.sum(np.asarray(vectors_tfidf.todense()),0)))[:100]

tfidf_mat=vectors_tfidf[:,top_tfidf].todense()


x_mat = tfidf_mat
y_mat = newsgroups_train.target
np.random.seed(0)

N_X=x_mat.shape[0]
n_train = int(round(0.8 * N_X))
p = np.random.permutation(N_X)
x_train, x_val = x_mat[p][:n_train,:], x_mat[p][n_train:,:] 
y_train, y_val = y_mat[p][:n_train], y_mat[p][n_train:] 


np.savetxt('in_data/news_train.txt',x_train)
np.savetxt('in_data/news_val.txt',x_val)
np.savetxt('in_data/news_lab_train.txt',y_train)
np.savetxt('in_data/news_lab_val.txt',y_val)


with open('in_data/words4.txt', 'w') as f:
  for word in feature_names[top_tfidf]:
    f.write(word+'\n')

with open('in_data/categories4.txt', 'w') as f:
  for cat in newsgroups_train.target_names:
    f.write(cat+'\n')

