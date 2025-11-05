import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import argparse
import torch.optim as optim

import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
import pandas as pd
import numpy as np
from sklearn import metrics
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt
import umap
import matplotlib
import mygene
from model import Mix_text_count_model
import pickle
import sklearn
import random
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# import sentence_transformers
plt.style.use('ggplot')
# plt.style.use('seaborn-v0_8-dark-palette')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
import matplotlib_inline

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, calinski_harabasz_score
import anndata as ad

try:
    import hnswlib

    hnswlib_imported = True
except ImportError:
    hnswlib_imported = False
    print("hnswlib not installed! We highly recommend installing it for fast similarity search.")
    print("To install it, run: pip install hnswlib")
from scipy.stats import mode


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result

def loss(a, b):
    temperature = 0.5

    a_norm = a
    b_norm = F.normalize(b, p=2, dim=1)
    similarity_matrix = torch.matmul(a_norm, b_norm.T)
    pos_sim = torch.diag(similarity_matrix)  # [N]
    # 创建mask，掩盖对角线位置
    mask = torch.eye(len(a), dtype=torch.bool, device=similarity_matrix.device)

    # 获取负样本相似度，排除正样本
    neg_sim_1 = similarity_matrix[~mask].reshape(len(a), -1)  # [N, N-1]
    neg_sim_2 = similarity_matrix.T[~mask].reshape(len(a), -1)  # [N, N-1]

    # 计算分子和分母
    exp_pos_sim = torch.exp(pos_sim / temperature)  # [N]
    exp_neg_sim_1 = torch.exp(neg_sim_1 / temperature).sum(dim=1)  # [N]
    exp_neg_sim_2 = torch.exp(neg_sim_2 / temperature).sum(dim=1)  # [N]

    # 计算InfoNCE Loss
    loss_1 = -torch.log(exp_pos_sim / exp_neg_sim_1 + 1e-8).mean()
    loss_2 = -torch.log(exp_pos_sim / exp_neg_sim_2 + 1e-8).mean()

    return (loss_1 + loss_2) / len(a)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def kldloss(p, q):
    c1 = -torch.sum(p * torch.log(q + 1e-8), dim=-1)
    c2 = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    return torch.mean(c1 - c2)


def train():
    model_mix.train()
    batchsize = 500
    dot_loss = 0
    mse_loss = 0
    kmeans_loss = 0
    text_emb_cat = []
    count_emb_cat = []
    for i in range(0, len(text), batchsize):
        length = min(len(text), batchsize + i)
        text_input = text[i:length].to('cuda:0')
        count_input = count[i:length]
        attention_mask_input = attention_mask[i:length]
        attention_mask_input_raw = attention_mask_raw[i:length]
        text_emb_batch, count_emb_batch, q_batch, pi, disp, mean, latent_batch = model_mix(text_input, count_input,
                                                                                           attention_mask_input,
                                                                                           attention_mask_input_raw)

        p = target_distribution(q_batch)

        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(count_input.squeeze(-1), mean, mean=True)

        dot_loss = dot_loss + loss(text_emb_batch, count_emb_batch) * 10
        mse_loss = mse_loss + zinb_loss * 10
        kmeans_loss = kmeans_loss + torch.mean(torch.sum(latent_batch, dim=1)) * 10


        text_emb_part = text_emb_batch.cpu().detach().numpy()
        count_emb_part = count_emb_batch.cpu().detach().numpy()

        text_emb_cat.append(text_emb_part)
        count_emb_cat.append(count_emb_part)

    text_emb = np.concatenate(text_emb_cat, axis=0)
    count_emb = np.concatenate(count_emb_cat, axis=0)

    (dot_loss + mse_loss + kmeans_loss).backward()

    optimizer.step()

    optimizer.zero_grad()
    return text_emb, count_emb, dot_loss, mse_loss, kmeans_loss


if __name__ == "__main__":
    def get_seq_embed_gpt(X, gene_names, prompt_prefix="", trunc_index=None):
        n_genes = X.shape[1]
        if trunc_index is not None and not isinstance(trunc_index, int):
            raise Exception('trunc_index must be None or an integer!')
        elif isinstance(trunc_index, int) and trunc_index >= n_genes:
            raise Exception('trunc_index must be smaller than the number of genes in the dataset')
        get_test_array = []
        for cell in (X):
            zero_indices = (np.where(cell == 0)[0])
            gene_indices = np.argsort(cell)[::-1]
            filtered_genes = gene_indices[~np.isin(gene_indices, list(zero_indices))]
            if trunc_index is not None:
                get_test_array.append(np.array(gene_names[filtered_genes])[0:trunc_index])
            else:
                get_test_array.append(np.array(gene_names[filtered_genes]))
        get_test_array_seq = [prompt_prefix + ' '.join(x) for x in get_test_array]
        return (get_test_array_seq)


    seed = 202310
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0')
    sampled_adata = sc.read_h5ad("./hPancreas/demo_train.h5ad")

    rna = sampled_adata.copy()

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=3000)
    rna = rna[:, rna.var.highly_variable]



    label = np.array(sampled_adata.obs['Celltype'])
    label_name = np.unique(label)
    label_encoder = LabelEncoder()
    label_encoder.fit(label_name)
    encoded_labels = label_encoder.transform(label)

    N_TRUNC_GENE = 1000

    sample_cells_data = get_seq_embed_gpt(np.array(sampled_adata.X), np.array(sampled_adata.var.index),
                                          prompt_prefix='A cell with genes ranked by expression: ',
                                          trunc_index=N_TRUNC_GENE)

    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True).to(device)

    batch_size = 100
    input_part = []
    attention_part = []
    for i in range(0, len(sample_cells_data), batch_size):
        length = min(len(sample_cells_data), batch_size + i)
        batch_inputs = sample_cells_data[i:length]
        encoded_input = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt').to(device)
        input_ids_part = encoded_input['input_ids']
        attention_mask_part = encoded_input['attention_mask']
        input_part.append(input_ids_part)
        attention_part.append(attention_mask_part)
    input_ids = torch.cat(input_part, dim=0)
    attention_mask = torch.cat(attention_part, dim=0)
    attention_mask_raw = attention_mask.clone()

    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * torch.finfo().min

    embbdings = model.embeddings(input_ids)
    embbdings = model.emb_drop(embbdings)
    text = model.emb_ln(embbdings).to('cpu')

    count = torch.tensor(np.array(rna.X)).unsqueeze(-1).float().to(device)

    model_mix = Mix_text_count_model(model, 768, 1, 1, 768, len(label_name), 0)
    model_mix.to(device)

    params_to_train = [param for name, param in model_mix.named_parameters() if 'model' not in name]
    optimizer = optim.SGD(params_to_train, lr=0.0035, weight_decay=5e-4)
    emb_text_max = []
    emb_count_max = []
    ari_max = 0
    nmi_max = 0
    params = []
    for epoch in range(20):
        text_emb, count_emb, dot_loss, mse_loss, kmeans_loss = train()
        print(' epoch: ', epoch, ' dot_loss = {:.2f}'.format(dot_loss), ' mse_loss = {:.2f}'.format(mse_loss),
              ' kmeans_loss = {:.2f}'.format(kmeans_loss))

        embedding_name = []
        cell_name = []
        for s in range(text_emb.shape[1]):
            embedding_name.append(str(s))

        for k in range(text_emb.shape[0]):
            cell_name.append(str(k))
        embedding_name = pd.DataFrame(index=embedding_name)
        cell_name = pd.DataFrame(index=cell_name)
        adata_learned = ad.AnnData(text_emb, obs=cell_name, var=embedding_name)
        adata_learned.obs['cell_type'] = label

        adata_learned.obs['batch'] = 1
        adata_learned.obs['batch'] = adata_learned.obs['batch'].astype("category")
        adata_learned.obsm['X_emb'] = text_emb

        import scib
        import scib.metrics as me

        embed = "X_emb"
        batch_key = "batch"
        label_key = "cell_type"
        cluster_key = "cluster"
        si_metric = "euclidean"
        subsample = 0.5
        verbose = False
        results = {}
        print('clustering...')
        res_max, nmi_max, nmi_all = scib.clustering.opt_louvain(adata_learned, label_key=label_key,
                                                                cluster_key=cluster_key, function=me.nmi,
                                                                use_rep=embed, verbose=verbose, inplace=True)

        nmi = me.nmi(adata_learned, group1=cluster_key, group2=label_key, method='arithmetic')

        ari = me.ari(adata_learned, group1=cluster_key, group2=label_key)


        if ari > ari_max:
            emb_text_max = text_emb
            emb_count_max = count_emb
            ari_max = ari
            nmi_max = nmi
            params = model_mix.state_dict()


    pd.DataFrame(emb_text_max).to_csv('./emb_text_train.csv')
    pd.DataFrame(emb_count_max).to_csv('./emb_count_train.csv')

    torch.save(params, './model_params.pth')
    del text_emb, count_emb, emb_text_max, emb_count_max




