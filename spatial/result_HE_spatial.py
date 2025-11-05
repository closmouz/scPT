import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tifffile import imread

# 1. 读入背景图
he_img = imread("ov_HE.tif")
adata=sc.read_h5ad('ov_final_spatial.h5ad')
print(adata)
# 2. 提取坐标 & 聚类结果
#coords = np.asarray(adata.obsm["spatial"])   # (n_obs, 2)


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

print(adata)

labels = adata.obs['leiden'].astype("category")
H, W = he_img.shape[:2]

plt.figure(figsize=(10, 10))
plt.imshow(he_img)

um_per_pixel=adata.uns['H&E resolution']
print(um_per_pixel)
x = coords[:, 0] / um_per_pixel
y = coords[:, 1] / um_per_pixel
scat = plt.scatter(
    x, y,
    c=labels.cat.codes,
    s=2, alpha=0.9, cmap="tab20", linewidths=0, rasterized=True
)

plt.axis("off")
plt.tight_layout()
plt.savefig("clusters_on_HE_res.png", dpi=300, bbox_inches="tight")
plt.close()
print("[save]", "clusters_on_HE_res.png")
