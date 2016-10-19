import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

target = 'fc5.csv'
df = pd.read_csv(os.path.join(os.path.dirname(__file__), target), index_col=0, header=None)
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

row_dendr = dendrogram(row_clusters, labels=df.index, orientation='right')
plt.show()
