import torch
import torch.nn as nn
import pdb, time
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean, scatter_add
from .mixformer3d import MultiScaleQuery
from torch.nn import functional as F

class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)): # batch_size
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
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
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


class QueryDecoder(BaseModule):
    """Query decoder for SPFormer.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, use_track_loss=False, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.use_track_loss = use_track_loss
        if self.use_track_loss:
            self.out_track_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'): # 是否有静态query
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i])) # [N_segments, 96] -> [N_segments, 256]
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


@MODELS.register_module()
class ScanNetMixQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, in_channels, share_attn_mlp, share_mask_mlp,
                 cross_attn_mode, mask_pred_mode, temporal_attn=False, bbox_flag=False, 
                 use_query_memory2=False, query_stage=[0],use_track_loss=False, use_temporal_loss=False, use_decouple=False, use_mot=False, mot_type=None, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, in_channels=in_channels,use_track_loss=use_track_loss, **kwargs)
        assert num_semantic_linears in [1, 2]
        assert isinstance(cross_attn_mode, list)
        assert isinstance(mask_pred_mode, list) 
        assert mask_pred_mode[-1] == "P"

        self.cross_attn_mode = cross_attn_mode
        self.mask_pred_mode = mask_pred_mode
        self.temporal_attn = temporal_attn

        self.share_attn_mlp = share_attn_mlp
        if not share_attn_mlp:
            if "P" in self.cross_attn_mode:
                self.input_pts_proj = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.share_mask_mlp = share_mask_mlp
        if not share_mask_mlp:
            if "P" in self.mask_pred_mode:
                self.x_pts_mask = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.ReLU(),
                    nn.Linear(d_model, d_model))

        self.bbox_flag = bbox_flag
        if self.bbox_flag:
            self.out_reg = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 6))

        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

        # ADDED
        self.use_query_memory2 = use_query_memory2
        if self.use_query_memory2:
            self.muti_scale_query = nn.ModuleList()
            self.muti_scale_norm = nn.ModuleList()
            self.query_stage = query_stage
            self.query_memory2 = [None for _ in range(len(self.query_stage))]
            self.pred_bbox_memory = [None for _ in range(len(self.query_stage))]
            for _ in range(len(self.query_stage)):
                self.muti_scale_query.append(MultiScaleQuery(embed_dims=256))
                self.muti_scale_norm.append(nn.LayerNorm(256))
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.merge_layer = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 256))
            # self.merge_layer = MergeLayer()
        self.use_mot = use_mot
        self.mot_type = mot_type
        if self.use_mot and mot_type == 'motr':
            self.motr_merge_layer = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 256))
        self.use_decouple = use_decouple
        if self.use_decouple:
            # self.query_inter = QueryInteractionX(256, 256)
            self.query_inter = MultiScaleQuery(embed_dims=256)
    def reset_decouple(self):
        """Reset the decouple module.
        """
        if self.use_decouple:
            self.before_query_memory = None
            self.before_query_boxes = None
    
    def reset_query_memory2(self):
        """Reset the detector.
        """
        if self.use_query_memory2:
            self.query_memory2 = [None for _ in range(len(self.query_stage))]
            self.pred_bbox_memory = [None for _ in range(len(self.query_stage))]

    def _forward_head(self, queries, mask_feats, mask_pts_feats, last_flag, layer):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, pred_bboxes = [], [], [], [], [], []
        object_queries = []
        for i in range(len(queries)): # batch_size
            norm_query = self.out_norm(queries[i]) # [N_segments, 256]
            object_queries.append(norm_query)
            cls_preds.append(self.out_cls(norm_query)) # 每个的类别 [N_segments, 256] —> [N_segments, 2(CA 200) or 19]
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag else None # None
            # pred_score = self.out_track_score(norm_query) if self.use_track_loss else pred_score
            pred_scores.append(pred_score) # 这个不存在
            if self.bbox_flag: # False (True in CA 200)
                reg_final = self.out_reg(norm_query) # [N_segments, 256] -> [N_segments, 6]
                reg_distance = torch.exp(reg_final[:, 3:6])
                pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            else: pred_bbox = None
            pred_bboxes.append(pred_bbox)
            if self.mask_pred_mode[layer] == "SP": # ['SP', 'SP', 'P', 'P']
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i]) # ? 计算点积 [N_q, N_seg] 为啥数量不一样, * (0.5 ~ 1) train的时候有query选择的部分
            elif self.mask_pred_mode[layer] == "P":
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_pts_feats[i]) # ? [61, 20000] 这个mask是完全基于特征相似度的吗？ 
            else:
                raise NotImplementedError("Query decoder not implemented!")
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool() # [N_q, N_seg] -> [N_q, N_seg]
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False # 如果所有键位置都被遮蔽，查询将无法与任何键进行交互，导致注意力权重无法正常计算。这可能会引发数值问题，如 NaN（非数字）值。
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, object_queries, pred_bboxes

    def forward_iter_pred(self, sp_feats, p_feats, queries, super_points, prev_queries=None, use_temporal_loss=False, inst_dict=None, track_instances=None, use_one2many=False):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        object_queries, pred_bboxes = [], []
        inst_feats = [self.input_proj(y) for y in sp_feats] if "SP" in self.cross_attn_mode else None # [N_segments, 96] -> [N_segments, 256]
        if use_temporal_loss and inst_dict is not None:
            before_features = inst_dict['querys']
            before_features_idx_list = inst_dict['best_obj_ids_list']
            # for i in range(len(inst_feats)):
            #     inst_feats[i][before_features_idx_list[i]] = self.merge_layer(torch.cat([inst_feats[i][before_features_idx_list[i]], before_features[i]], dim=1))

            for i in range(len(inst_feats)):
                # 取出原来的 inst_feats[i]
                old_feats = inst_feats[i]
                # 计算需要替换进去的 merged_part
                merged_part = self.merge_layer(
                    torch.cat([old_feats[before_features_idx_list[i]], before_features[i]], dim=1)
                )

                # 克隆出一个新的张量
                new_feats = old_feats.clone()
                # 在克隆的张量上进行索引赋值
                new_feats[before_features_idx_list[i]] = merged_part

                # 将更新后的张量放回 inst_feats[i]
                inst_feats[i] = new_feats
        if track_instances is not None:
            pass

        inst_pts_feats = [self.input_proj(y) if self.share_attn_mlp else self.input_pts_proj(y)
             for y in p_feats] if "P" in self.cross_attn_mode else None # None
        mask_feats = [self.x_mask(y) for y in sp_feats] if "SP" in self.mask_pred_mode else None # [N_segments, 96] -> [N_segments, 256]
        mask_pts_feats = [self.x_mask(y) if self.share_mask_mlp else self.x_pts_mask(y)
             for y in p_feats] if "P" in self.mask_pred_mode else None # [20000, 99] -> [20000, 256]
        queries = self._get_queries(queries, len(sp_feats)) # [N_segments, 96] -> [N_segments, 256] # ! 这个和inst_feats差不多？
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
             self._forward_head(queries, mask_feats, mask_pts_feats, last_flag=False, layer=0)
        cls_preds.append(cls_pred) # [N_segments, 2]
        sem_preds.append(sem_pred) # None
        pred_scores.append(pred_score) # None
        pred_masks.append(pred_mask) # [N_segments, N_segments]
        object_queries.append(object_query) # [N_segments, 256]
        pred_bboxes.append(pred_bbox) # [N_segments, 6]

            
        if use_one2many:
            one2many_output_dict = {'cls_preds':[], 'sem_preds':[], 'scores':[], 'masks':[], 'queries':[], 'pred_bboxes':[]}
            one2many_output_dict['cls_preds'].append(cls_pred)
            one2many_output_dict['sem_preds'].append(sem_pred)
            one2many_output_dict['scores'].append(pred_score)
            one2many_output_dict['masks'].append(pred_mask)
            one2many_output_dict['queries'].append(object_query)
            one2many_output_dict['pred_bboxes'].append(pred_bbox)
            one2many_queries= []
            one2many_attn_mask = []
            for i in range(len(queries)):
                one2many_queries.append(queries[i].clone())
                one2many_attn_mask.append(attn_mask[i].clone())

        for i in range(len(self.cross_attn_layers)): # 3 [queries查询inst_feats] [queries查询inst_feats] [queries查询inst_feats]
            if self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "SP": # SP 内的attention，使用mask
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask) # K Q mask [N_segments, 256], [N_segments, 256], [N_segments, N_segments] -> [N_segments, 256]
                if use_one2many:
                    one2many_queries = self.cross_attn_layers[i](inst_feats, one2many_queries, one2many_attn_mask)
            elif self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "P":   # current method, change P mask to SP
                if not use_temporal_loss:
                    xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0) # torch.chunk(input, chunks, dim=0) 会把输入的张量 input 按照指定的维度 dim 和指定的分块数 chunks 来分割。
                    attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1) # [20000, 1]
                        for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights)] # 将点的mask转换为超点的mask
                    attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan 所有最小的attn对应的都设定为False
                    for j in range(len(attn_mask)): # batch_size
                        mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        attn_mask[j] *= mask
                else:
                    # attn_mask = None
                    xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0) # torch.chunk(input, chunks, dim=0) 会把输入的张量 input 按照指定的维度 dim 和指定的分块数 chunks 来分割。
                    attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1) # [20000, 1]
                        for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights)] # 将点的mask转换为超点的mask
                    attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan 所有最小的attn对应的都设定为False
                    for j in range(len(attn_mask)): # batch_size
                        mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        attn_mask[j] *= mask
                    
                    # if attn_mask[0].shape[1] != inst_feats[0].shape[0]:
                    #     attn_mask = [torch.cat([att, torch.zeros(att.shape[0], inst_feats[0].shape[0] - att.shape[1], device=att.device).bool()], dim=1) for att in attn_mask]
                    for j in range(len(attn_mask)):
                        if attn_mask[j].shape[1] != inst_feats[j].shape[0]:
                            attn_mask[j] = torch.cat([attn_mask[j], torch.zeros(attn_mask[j].shape[0], inst_feats[j].shape[0] - attn_mask[j].shape[1], device=attn_mask[j].device).bool()], dim=1)
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                if use_one2many:
                    one2many_xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0) # torch.chunk(input, chunks, dim=0) 会把输入的张量 input 按照指定的维度 dim 和指定的分块数 chunks 来分割。
                    one2many_attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1) # [20000, 1]
                        for att, sp, xyz_w in zip(one2many_attn_mask, super_points[0], one2many_xyz_weights)] # 将点的mask转换为超点的mask
                    one2many_attn_mask = [(att > 0.5).bool() for att in one2many_attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan 所有最小的attn对应的都设定为False
                    for j in range(len(one2many_attn_mask)): # batch_size
                        mask = ~(one2many_attn_mask_score[j] == one2many_attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        one2many_attn_mask[j] *= mask
                    one2many_queries = self.cross_attn_layers[i](inst_feats, one2many_queries, one2many_attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "SP":
                attn_mask = [att[:, sp] for att, sp in zip(attn_mask, super_points[0])]
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "P":
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            else:
                raise NotImplementedError("Not support yet!")

            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            if use_one2many:
                # one2many_queries = self.self_attn_layers[i](one2many_queries)
                one2many_queries = self.ffn_layers[i](one2many_queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            if self.use_decouple and i == 2:
                if self.before_query_memory is not None:
                    queries = self.query_inter([pred_bbox[j][:, :3] for j in range(len(pred_bbox))], queries, self.before_query_memory, self.before_query_memory, self.before_query_boxes)
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
                 self._forward_head(queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)
            if use_one2many:
                one2many_cls_pred, one2many_sem_pred, one2many_pred_score, one2many_pred_mask, one2many_attn_mask, one2many_object_query, one2many_pred_bbox = \
                        self._forward_head(one2many_queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)
                # one2many_output = dict(
                #     cls_preds=one2many_cls_pred, sem_preds=one2many_sem_pred, masks=one2many_pred_mask, scores=one2many_pred_score,
                #     queries=one2many_object_query, bboxes=one2many_pred_bbox)
                one2many_output_dict['cls_preds'].append(one2many_cls_pred)
                one2many_output_dict['sem_preds'].append(one2many_sem_pred)
                one2many_output_dict['scores'].append(one2many_pred_score)
                one2many_output_dict['masks'].append(one2many_pred_mask)
                one2many_output_dict['queries'].append(one2many_object_query)
                one2many_output_dict['pred_bboxes'].append(one2many_pred_bbox)
            if self.use_decouple and i == 2:
                self.before_query_memory = [queries[j].clone() for j in range(len(queries))]
                self.before_query_boxes = [pred_bbox[j][:, :3].clone()  for j in range(len(pred_bbox))]
            if self.use_query_memory2 and i in self.query_stage:
                if self.query_memory2[i] is not None:
                    queries_new = self.muti_scale_query[i]([pred_bbox[j][:, :3] for j in range(len(pred_bbox))], queries, self.query_memory2[i], self.query_memory2[i], self.pred_bbox_memory[i])
                    queries = [queries[j] + queries_new[j] for j in range(len(queries))]
                    queries = [self.muti_scale_norm[i](queries[j]) for j in range(len(queries))]
                with torch.no_grad():
                    detach_query = [queries[j].clone().detach() for j in range(len(queries))]
                    detach_pred_bbox = [pred_bbox[j][:, :3].clone().detach() for j in range(len(pred_bbox))]
                self.query_memory2[i] = detach_query
                self.pred_bbox_memory[i] = detach_pred_bbox
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            object_queries.append(object_query)
            pred_bboxes.append(pred_bbox)
        
        neq_sum_list = []
        for cls_pred in cls_preds[:-1]:
            cls_name = cls_pred[0].argmax(-1, keepdim=True).flatten()
            neq_sum = (cls_name != cls_preds[-1][0].argmax(-1, keepdim=True).flatten()).sum(-1)
            neq_sum_list.append(neq_sum.item())
        # print(neq_sum_list)
        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores, bboxes=bboxes)
            for cls_pred, sem_pred, scores, masks, bboxes in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1], pred_bboxes[:-1])]
        if use_one2many:
            one2many_output_aux_outputs = [
                dict(
                    cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores, bboxes=bboxes)
                for cls_pred, sem_pred, scores, masks, bboxes in zip(
                    one2many_output_dict['cls_preds'][:-1], one2many_output_dict['sem_preds'][:-1], one2many_output_dict['scores'][:-1], one2many_output_dict['masks'][:-1], one2many_output_dict['pred_bboxes'][:-1])
            ]
            one2many_outputs = dict(
                cls_preds=one2many_output_dict['cls_preds'][-1],
                sem_preds=one2many_output_dict['sem_preds'][-1],
                masks=one2many_output_dict['masks'][-1],
                scores=one2many_output_dict['scores'][-1],
                queries=one2many_output_dict['queries'][-1],
                bboxes=one2many_output_dict['pred_bboxes'][-1],
                aux_outputs=one2many_output_aux_outputs)
            return dict(
                cls_preds=cls_preds[-1],
                sem_preds=sem_preds[-1],
                masks=pred_masks[-1],
                scores=pred_scores[-1],
                queries=object_queries[-1],
                bboxes=pred_bboxes[-1],
                aux_outputs=aux_outputs,
                one2many_outputs=one2many_outputs,)
        else:
            return dict(
                cls_preds=cls_preds[-1],
                sem_preds=sem_preds[-1],
                masks=pred_masks[-1],
                scores=pred_scores[-1],
                queries=object_queries[-1],
                bboxes=pred_bboxes[-1],
                aux_outputs=aux_outputs)
    
    def forward(self, sp_feats, p_feats, queries, super_points, prev_queries=None, use_temporal_loss=False, inst_dict=False,track_instances=None, use_one2many=False):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(sp_feats, p_feats, queries, super_points, prev_queries, use_temporal_loss=use_temporal_loss, inst_dict=inst_dict, track_instances=track_instances, use_one2many=use_one2many)
        else:
            raise NotImplementedError("No simple forward!!!")


@MODELS.register_module()
class S3DISQueryDecoder(QueryDecoder):
    # Does it have any differences with QueryDecoder?
    pass


class MergeLayer(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=256, dropout=0.0):
        super(MergeLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 增加LayerNorm来稳定训练
            nn.ReLU(),
            # nn.Dropout(dropout),       # 若需要，使用Dropout提高泛化
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),     # 再次进行LayerNorm
            nn.ReLU(),
            # nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.layer(x)
class FFN2(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt
class QueryInteractionX(nn.Module):
    def __init__(self, in_channels, mid_channels, **kwargs):
        super().__init__()
        dropout = kwargs.get('drop_rate', 0.0)
        self.with_att = kwargs.get('with_att', True)
        self.with_pos = kwargs.get('with_pos', True)
        self.self_attn = nn.MultiheadAttention(in_channels, 8, dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN2(in_channels, mid_channels, dropout)

    def forward(self, track_embed, obj_embed, pos_embed=None):
        track_num = len(track_embed)
        query_embed = torch.cat([track_embed, obj_embed], dim=0)
        
        # add position embedding
        if self.with_pos and pos_embed is not None:
            pos_embed = torch.cat([pos_embed, pos_embed], dim=0)
            q = k = query_embed + pos_embed
        else:
            q = k = query_embed

        tgt = query_embed.clone()
        # attention
        if self.with_att:
            tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        track_embed = tgt[:track_num]
        obj_embed = tgt[track_num:]

        return track_embed, obj_embed