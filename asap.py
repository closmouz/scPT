import sys, os

import numpy as np
from umap import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

data_dir = "./ASAP-PBMC/"

genes = pd.read_csv(data_dir + "genes.txt", header = None).values.squeeze()
counts_rna1 = np.array(sp.load_npz(os.path.join(data_dir, "GxC1.npz")).todense().T)
counts_rna2 = np.array(sp.load_npz(os.path.join(data_dir, "GxC2.npz")).todense().T)
counts_rnas = [counts_rna1, counts_rna2]


labels1=pd.read_csv(os.path.join(data_dir, 'meta_c' + str(1) + '.csv'), index_col=0)["coarse_cluster"].values.squeeze()
labels2=pd.read_csv(os.path.join(data_dir, 'meta_c' + str(2) + '.csv'), index_col=0)["coarse_cluster"].values.squeeze()

embedding_name1 = []
cell_name1 = []
for s in range(counts_rna1.shape[1]):
    embedding_name1.append(str(s))

for k in range(counts_rna1.shape[0]):
    cell_name1.append(str(k))
embedding_name1 = pd.DataFrame(index=embedding_name1)
cell_name1 = pd.DataFrame(index=cell_name1)
adata_learned1 = ad.AnnData(counts_rna1, obs=cell_name1, var=embedding_name1)
adata_learned1.obs['cell_type'] = labels1
adata_learned1.var['gene_name'] = genes
adata_learned1.obs['batch'] = 1


embedding_name2 = []
cell_name2 = []
for s in range(counts_rna2.shape[1]):
    embedding_name2.append(str(s))

for k in range(counts_rna2.shape[0]):
    cell_name2.append(str(k))
embedding_name2 = pd.DataFrame(index=embedding_name2)
cell_name2 = pd.DataFrame(index=cell_name2)
adata_learned2 = ad.AnnData(counts_rna2, obs=cell_name2, var=embedding_name2)
adata_learned2.obs['cell_type'] = labels2
adata_learned2.var['gene_name'] = genes
adata_learned2.obs['batch'] = 2

print(adata_learned1)
print(adata_learned2)

import scanpy as sc
adata = sc.concat([adata_learned1, adata_learned2], merge='same')  # 沿着观测（cells）维度合并

adata.write('./ASAP-PBMC/rna.h5ad')
