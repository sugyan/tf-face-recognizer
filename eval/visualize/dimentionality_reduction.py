import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

x, y = load_svmlight_file('fc6.txt')
x = StandardScaler().fit_transform(x.toarray())

def pca(x):
    pca = PCA(n_components=2, whiten=False)
    return pca.fit_transform(x)

def t_sne(x, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        init='pca',
        method='exact')
    return tsne.fit_transform(x)

def draw(p):
    colors = {
        1: 'green',
        2: 'pink',
        3: 'red',
        4: 'yellow',
        5: 'purple'
    }
    for i in range(len(p)):
        color = colors[y[i]]
        plt.scatter(*p[i], c=color, s=30)

# PCA
plt.clf()
plt.title('PCA')
draw(pca(x))
plt.savefig('PCA.png')

# t-SNE
plt.clf()
plt.title('t-SNE')
draw(t_sne(x, perplexity=30, early_exaggeration=2.0, learning_rate=100))
plt.savefig('TSNE.png')
