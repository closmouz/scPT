import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
import anndata as ad
import os
from sklearn import metrics
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from sklearn.metrics import silhouette_score,adjusted_rand_score,homogeneity_score,normalized_mutual_info_score,adjusted_mutual_info_score,calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix




all_embs = []
rna = sc.read_h5ad("spatial.h5ad")
rna.obs['annotation'] = (
    rna.obs['annotation']
    .astype(object)            # 确保可以填字符串
    .fillna("unknown")         # 把 NaN 填成 'unknown'
    .astype("category")        # 重新转成分类
)
print(rna.obs['annotation'])
datasize=10000
for j in range(0, len(rna), datasize):
    path= str(j)+'_emb_text_test.csv'
    emb=pd.read_csv(path).values[:,1:]
    all_embs.append(emb)

all_embs = np.vstack(all_embs)

rna.X=np.array(rna.X.todense())
print(rna)

rna.obs['cell_type']=rna.obs['annotation']
rna.obs['batch']=1
rna.obs['batch']=rna.obs['batch'].astype("category")
rna.obsm['X_emb']=all_embs

sc.pp.neighbors(rna, use_rep='X_emb')




import scib
import scib.metrics as me
embed = "X_emb"
batch_key = "batch"
label_key = "cell_type"
cluster_key = "cluster"
si_metric = "euclidean"
subsample = 0.5
verbose = False
results={}
print('clustering...')

sc.tl.louvain(rna, resolution=0.4, key_added='cluster')
print(rna.obs['cluster'])

pd.DataFrame(rna.obs).to_csv('obs0.4.csv')



results['NMI'] = me.nmi(rna, group1=cluster_key, group2=label_key, method='arithmetic')
print("NMI: " + str(results['NMI']))

results['ARI'] = me.ari(rna, group1=cluster_key, group2=label_key)
print("ARI: " + str(results['ARI']))


results['ASW'] = me.silhouette(rna,label_key=label_key,embed=embed)
print("ASW: " + str(results['ASW']))


sc.tl.umap(rna)

sc.pl.umap(rna,color=['cell_type'],save='_ov_emb_umap.pdf')
