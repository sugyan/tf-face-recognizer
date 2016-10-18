import os
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

target = 'fc5.csv'
df = pd.read_csv(os.path.join(os.path.dirname(__file__), target), index_col=0, header=None)
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

cl0 = row_clusters[0]
print(df.index[cl0[0]], df.index[cl0[1]])
