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



path=''
emb=pd.read_csv(path).values[:,1:]
rna=sc.read_h5ad('./hPancreas/demo_train.h5ad')
rna.X=np.array(rna.X)

embedding_name = []
cell_name = []
for s in range(emb.shape[1]):  
    embedding_name.append(str(s))

for k in range(emb.shape[0]):
    cell_name.append(str(k))
embedding_name = pd.DataFrame(index=embedding_name)

cell_name = pd.DataFrame(index=cell_name)
adata = ad.AnnData(emb, obs=cell_name, var=embedding_name)
adata.obs['cell_type'] =np.array(rna.obs['Celltype'])

groud_true=adata.obs['cell_type'].copy()
adata.obs['cell_type']=adata.obs['cell_type'].astype("category")
print(adata.obs['cell_type'])
adata.obs['batch']=1
adata.obs['batch']=adata.obs['batch'].astype("category")
adata.obsm['X_emb']=emb
sc.pp.neighbors(adata, use_rep='X')




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
res_max, nmi_max, nmi_all = scib.clustering.opt_louvain(adata, label_key=label_key,cluster_key=cluster_key, function=me.nmi, use_rep=embed, verbose=verbose, inplace=True)


results['NMI'] = me.nmi(adata, group1=cluster_key, group2=label_key, method='arithmetic')
print("NMI: " + str(results['NMI']))

results['ARI'] = me.ari(adata, group1=cluster_key, group2=label_key)
print("ARI: " + str(results['ARI']))


results['ASW'] = me.silhouette(adata,label_key=label_key,embed=embed)
print("ASW: " + str(results['ASW']))



sc.tl.umap(adata)

sc.pl.umap(adata,color=['cell_type'],save='umap.pdf')

import json
from pySankey.sankey import sankey
with open('cell_type_colors.json', 'r', encoding='utf-8') as file:
    color = json.load(file)

pl.figure()

index=np.array(adata.obs['cluster'])
type=np.array(adata.obs['cell_type'])

df = pd.DataFrame({'index': index, 'type': type})

# 按type的总数量排序
counts_total1 = df['type'].value_counts()
sorted_types = counts_total1.index.tolist()


counts_total2 = df['index'].value_counts()
sorted_index = counts_total2.index.tolist()

df['type'] = pd.Categorical(df['type'], categories=sorted_types, ordered=True)
df['index'] = pd.Categorical(df['index'], categories=sorted_index, ordered=True)

sankey(left=index,right=type,leftLabels=list(reversed(sorted_index)),rightLabels=list(reversed(sorted_types)), aspect=40, colorDict=color, fontsize=6)
pl.savefig('./figures/umap/sankey.pdf')







