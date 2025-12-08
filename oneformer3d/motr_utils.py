from typing import List, Optional
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from scipy.optimize import linear_sum_assignment
class SelfAttention(nn.Module):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self,in_channels=96,d_model=256,num_heads=8,dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.update_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

    
    @staticmethod
    def with_pos_embed(tensor, last_queries):
        return tensor if last_queries is None else tensor + last_queries
    
    def _forward_track_attn(self, tgt, firstframenum):
        # q = k = self.with_pos_embed(tgt, last_queries)
        q = k = tgt # [batch_size, seq_len, feature_dim]
        if q.shape[1] > firstframenum: #如果查询序列的长度大于300，代码会对后半部分（q[:, 300:]）的特征进行更新。
            tgt2, _= self.attn(q[:, firstframenum:], #.transpose(0, 1) 是对张量的转置操作，交换维度0和维度1，通常是为了匹配自注意力机制的输入格式（通常是 [batch_size, seq_len, feature_dim]）。
                                    k[:, firstframenum:],
                                    tgt[:, firstframenum:])
            tgt = torch.cat([tgt[:, :firstframenum],self.norm4(tgt[:, firstframenum:]+self.dropout5(tgt2[0]))], dim=1) #将更新后的特征与原特征融合
        return tgt
    
    def forward(self, x, firstnum):
        # [bs(1),N,96]->[N,96]
        x = x.unsqueeze(0)
        # if pre_queries : 这么写不适用于tensor
        # if isinstance(pre_queries, torch.Tensor):
        x = self._forward_track_attn(x, firstnum)
        z, _ = self.attn(x, x, x)
        z = self.dropout(z) + x
        z = self.norm(z)
        # else:
            # out = []
            # for y in x:
            # z, _ = self.attn(x, x, x)
            # z = self.dropout(z) + x
            # z = self.norm(z)
        return z

class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model=256, hidden_dim=1024, dropout=0.0, activation_fn='gelu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        # out = []
        # for y in x:
        z = self.net(x)
        z = z + x
        z = self.norm(z)
            # out.append(z)
        return z

def compute_cost_matrix(track_query_feats, obj_query_feats, track_query_pos, obj_query_pos, alpha=1.0, beta=1.0):
    # 特征距离 (cosine 或 L2)
    feat_cost = torch.cdist(track_query_feats, obj_query_feats, p=2)  # (N_A, N_B)
    
    # 位置距离
    pos_cost = torch.cdist(track_query_pos, obj_query_pos, p=2)  # (N_A, N_B)

    # 综合 cost
    cost_matrix = alpha * feat_cost + beta * pos_cost
    return cost_matrix


def sinkhorn(log_alpha, n_iters=20):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)

def gumbel_sinkhorn(log_alpha, temp=1.0, noise=True, n_samples=1, n_iters=20):
    if noise:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_alpha)))
        log_alpha = log_alpha + gumbel_noise
    log_alpha = log_alpha / temp
    return sinkhorn(log_alpha, n_iters)

def match_queries(
    track_query_feats, obj_query_feats,
    track_query_pos, obj_query_pos,
    match_method='hungarian',
    alpha=1.0, beta=1.0,
    temperature=0.1,
    n_iters=50
):
    cost_mat = compute_cost_matrix(track_query_feats, obj_query_feats, track_query_pos, obj_query_pos, alpha, beta)  # [N, M]

    if match_method == 'hungarian':
        # 硬匹配
        row_ind, col_ind = linear_sum_assignment(cost_mat.detach().cpu().numpy())
        best_obj_ids = torch.full((track_query_feats.shape[0],), -1, dtype=torch.long, device=cost_mat.device)
        best_obj_ids[row_ind] = torch.tensor(col_ind, device=cost_mat.device)
        return best_obj_ids

    elif match_method == 'gumbel': # Learning Latent Permutations with Gumbel-Sinkhorn Networks
        # 可微匹配
        perm_mat = gumbel_sinkhorn(-cost_mat[None], temp=temperature, n_iters=n_iters)[0]  # (N, M)
        matched = torch.matmul(perm_mat, obj_query_feats)  # [N, C] 软融合后的特征，或者使用argmax来近似hard assignment
        best_obj_ids = perm_mat.argmax(dim=-1)  # 可选，近似hard match
        return best_obj_ids

    else:
        raise ValueError(f"Unsupported match method: {match_method}")
