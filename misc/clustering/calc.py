import os
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

target = 'fc5.csv'
df = pd.read_csv(os.path.join(os.path.dirname(__file__), target), header=None, dtype={ 0: str }).set_index(0)
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

for row in row_clusters:
    print(row)
