import sys
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

if len(sys.argv) < 2:
    print('usage: {} <csv file>'.format(sys.argv[0]))
    sys.exit()
target = sys.argv[1]
df = pd.read_csv(target, header=None, dtype={0: str}).set_index(0)
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

for row in row_clusters[:30]:
    if row[3] > 2:
        break
    print(df.index[int(row[0])], df.index[int(row[1])], row[2])
