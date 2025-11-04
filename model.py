import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        #self.attn = nn.MultiheadAttention(d_model, n_head,dropout=0.3)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        x = x.float()
        #print(x.dtype)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

#(width=512,layers=1,heads=1,output_dim=512)
class VisionTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()

        self.output_dim = output_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # x shape = [*, width, 1]

        x = x.permute(0, 2, 1)  # shape = [*, 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, 2, width]

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x=x.float()
            x = x @ self.proj

        return x

class StackedVisionTransformer(nn.Module):
    def __init__(self, num_vit: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()

        # 创建 12 个 VisionTransformer 并将它们放在一个 ModuleList 中
        self.vit_stack = nn.ModuleList(
            [VisionTransformer(width, layers, heads, output_dim) for _ in range(num_vit)]
        )

    def forward(self, x: torch.Tensor):
        # 遍历每个 VisionTransformer，将当前模块的输出作为下一个模块的输入
        for vit in self.vit_stack:
            x = vit(x)  # 每个 ViT 模块的输出是下一个模块的输入

        return x


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




def cal_latent(z):
    sum_y = torch.sum(torch.square(z), dim=1)
    num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y    
    num = num 
    num = torch.pow(1.0 + num, -(1 + 1.0) / 2.0)
    zerodiag_num = num - torch.diag(torch.diag(num))
    latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
    return num, latent_p



def cal_dist(z, clusters):
    # 计算距离 dist1
    dist1 = torch.sum(torch.square(z.unsqueeze(1) - clusters), dim=2)
    
    # 距离归一化
    temp_dist1 = dist1 - torch.reshape(torch.min(dist1, dim=1, keepdim=True)[0], [-1, 1])
    
    # 计算 q
    q = torch.exp(-temp_dist1)
    
    # q 归一化
    q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0, 1)
    
    # q 的平方
    q = torch.pow(q, 2)
    
    # 再次归一化 q
    q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0, 1)
    
    # 计算加权距离 dist2
    dist2 = dist1 * q
    
    return dist1, dist2




class Mix_text_count_model(nn.Module):
    def __init__(self, model, width: int, layers: int, heads: int, output_dim: int, label_number, dropout):
        super().__init__()
        self.vit_stack = nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(output_dim, output_dim),
                LayerNorm(output_dim),
                torch.nn.Linear(width, output_dim)
            )
            for _ in range(11)
        ])
        self.first = torch.nn.Sequential(
            torch.nn.Linear(3000, width),
            torch.nn.BatchNorm1d(width),
            torch.nn.ReLU()
        )
        self.vit_stack_0 = torch.nn.Sequential(
            torch.nn.Linear(3000, output_dim),
            LayerNorm(output_dim),
            torch.nn.Linear(width, output_dim)
        )

        self.count_13 = torch.nn.Sequential(
            torch.nn.Linear(output_dim, output_dim),
            LayerNorm(output_dim),
            torch.nn.Linear(width, output_dim)
        )
        scale = width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.model = model
        self.last = torch.nn.Sequential(
            torch.nn.Linear(23185, width),
            torch.nn.BatchNorm1d(width),
            torch.nn.ReLU()
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(label_number, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(768, 1500),
            torch.nn.BatchNorm1d(1500),
            torch.nn.ReLU()
        )
        # 无
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Linear(1500, 3000),
            torch.nn.BatchNorm1d(3000),
            torch.nn.ReLU()
        )
        self.dropout = dropout
        self.pi = torch.nn.Linear(1500, 3000)
        self.disp = torch.nn.Linear(1500, 3000)
        self.mean = torch.nn.Linear(1500, 3000)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

        self.classify = nn.Linear(768, label_number)

        self.clusters = clusters = nn.Parameter(torch.empty(label_number, 768))  # 创建一个空的张量

        init.xavier_uniform_(clusters)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attention_mask, attention_mask_raw):
        # 遍历每个 VisionTransformer，将当前模块的输出作为下一个模块的输入
        # print(x.size())
        # print(attention_mask.size())
        # print(y.squeeze(-1).size(),'count_input')
        # 改3000直接到隐层
        # y=self.first(y.squeeze(-1))
        # print(y.size())
        count = self.vit_stack_0(y.squeeze(-1))
        text = torch.cat([x, count.unsqueeze(1)], dim=1)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 1), value=0)
        attention_mask_raw = torch.nn.functional.pad(attention_mask_raw, (0, 1), value=1)
        with torch.no_grad():
            x = self.model.encoder.layers[0](hidden_states=text, hidden_states2=text, attention_mask=attention_mask)[0]

        # print(attention_mask.size(),'attention_mask_size')
        for i in range(10):
            # Sprint(y,i)
            count = self.vit_stack[i](count.squeeze(-1))
            text = torch.cat([x, count.unsqueeze(1)], dim=1)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, 1), value=0)
            attention_mask_raw = torch.nn.functional.pad(attention_mask_raw, (0, 1), value=1)
            with torch.no_grad():
                x = self.model.encoder.layers[i + 1](hidden_states=text, hidden_states2=text,
                                                     attention_mask=attention_mask)[0]
        count = self.vit_stack[10](count.squeeze(-1))
        text = torch.cat([x, count.unsqueeze(1)], dim=1)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 1), value=0)
        attention_mask_raw = torch.nn.functional.pad(attention_mask_raw, (0, 1), value=1)

        with torch.no_grad():
            x = self.model.encoder.layers[11](hidden_states=text, hidden_states2=text, attention_mask=attention_mask)[0]
        attention_mask_raw = attention_mask_raw.squeeze()
        x = mean_pooling(x, attention_mask_raw)
        text_emb = F.normalize(x, p=2, dim=1)
        count_1 = self.count_13(count.squeeze(-1))
        count_emb = count_1 @ self.proj

        _, q = cal_latent(text_emb)
        z = self.decoder1(count_1)
        z = F.dropout(z, self.dropout, training=self.training)
        pi = torch.sigmoid(self.pi(z))
        disp = self.DispAct(self.disp(z))
        mean = self.MeanAct(self.mean(z))

        _, latent = cal_dist(text_emb, self.clusters)

        return text_emb, count_emb, q, pi, disp, mean, latent





