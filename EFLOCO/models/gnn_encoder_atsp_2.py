import torch
import torch.nn.functional as F
from torch import nn
import functools
import math
from models.nn import (  
  linear,
  zero_module,
  normalization,
  timestep_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint


try:
  from torch_sparse import SparseTensor
except ImportError:
  SparseTensor = None



class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, graph, mode="residual", edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          graph: torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    if not sparse:
      batch_size, num_nodes, hidden_dim = h.shape
    else:
      batch_size = None
      num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e

    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H

    if not sparse:
      Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    if not sparse:
      e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
    else:
      e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # Update node features
    h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

    # Normalize node features
    if not sparse:
      h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
    else:
      h = self.norm_h(h) if self.norm_h else h

    # Normalize edge features
    if not sparse:
      e = self.norm_e(
          e.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e

    return h, e

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if not sparse:
      if (mode or self.aggregation) == "mean":
        return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
      elif (mode or self.aggregation) == "max":
        return torch.max(Vh, dim=2)[0]
      else:
        return torch.sum(Vh, dim=2)
    # else:
      # sparseVh = SparseTensor(
      #     row=edge_index[0],
      #     col=edge_index[1],
      #     value=Vh,
      #     sparse_sizes=(graph.size(0), graph.size(1))
      #     )

      # if (mode or self.aggregation) == "mean":
      #   return sparse_mean(sparseVh, dim=1)
      #
      # elif (mode or self.aggregation) == "max":
      #   return sparse_max(sparseVh, dim=1)
      #
      # else:
      #   return sparse_sum(sparseVh, dim=1)



class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x


class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x


def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward
class GNNEncoder(nn.Module):

  def __init__(self, n_layers, hidden_dim, num_nodes=100, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()

    self.sparse = sparse
    if self.sparse and SparseTensor is None:
      raise ImportError("torch_sparse is not installed. Please install it to use sparse mode.")

    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2

    # self.node_feature_embedder = nn.Sequential(
    #   nn.Linear(2 * num_nodes, hidden_dim * 2),
    #   nn.ReLU(),
    #   nn.Linear(hidden_dim * 2, hidden_dim)
    # )


    # self.node_feature_embedder = nn.Sequential(
    #   nn.Linear(2, hidden_dim * 2),
    #   nn.ReLU(),
    #   nn.Linear(hidden_dim * 2, hidden_dim)
    # )

    self.node_feature_embedder = nn.Sequential(
      nn.Linear(8, hidden_dim * 2),
      nn.ReLU(),
      nn.Linear(hidden_dim * 2, hidden_dim)
    )




    self.edge_cost_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    self.time_embed = nn.Sequential(
      linear(hidden_dim, time_embed_dim),
      nn.ReLU(),
      linear(time_embed_dim, time_embed_dim),
    )
    self.out = nn.Sequential(
      normalization(hidden_dim),
      nn.ReLU(),
      nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
    )
    self.layers = nn.ModuleList([
      GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
      for _ in range(n_layers)
    ])
    self.time_embed_layers = nn.ModuleList([
      nn.Sequential(nn.ReLU(), linear(time_embed_dim, hidden_dim))
      for _ in range(n_layers)
    ])
    self.per_layer_out = nn.ModuleList([
      nn.Sequential(nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm), nn.SiLU(),
                    zero_module(nn.Linear(hidden_dim, hidden_dim)))
      for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self, cost_matrix, timesteps, xt):

    B, N, _ = cost_matrix.shape


    # outgoing_costs = cost_matrix
    # incoming_costs = cost_matrix.transpose(1, 2)
    # node_features_raw = torch.cat([outgoing_costs, incoming_costs], dim=-1)  # (B, N, 2N)
    # x = self.node_feature_embedder(node_features_raw.view(B * N, -1)).view(B, N, -1)  # (B, N, H)


    # node_out_feat = cost_matrix.mean(dim=2, keepdim=True)  # (B, N, 1)
    # node_in_feat = cost_matrix.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, N, 1)
    # node_feat = torch.cat([node_out_feat, node_in_feat], dim=-1)  # (B, N, 2)

    # duo tezheng fanhua
    node_feat = torch.cat([

      cost_matrix.mean(dim=-1, keepdim=True),
      cost_matrix.std(dim=-1, keepdim=True),
      cost_matrix.max(dim=-1, keepdim=True).values,
      cost_matrix.min(dim=-1, keepdim=True).values,

      cost_matrix.transpose(1, 2).mean(dim=-1, keepdim=True),
      cost_matrix.transpose(1, 2).std(dim=-1, keepdim=True),
      cost_matrix.transpose(1, 2).max(dim=-1, keepdim=True).values,
      cost_matrix.transpose(1, 2).min(dim=-1, keepdim=True).values
    ], dim=-1)  

    x = self.node_feature_embedder(node_feat)  # (B, N, H)


    e_from_cost = self.edge_embed(self.edge_cost_embed(cost_matrix))
    e_from_xt = self.edge_embed(self.edge_cost_embed(xt.float()))
    e = e_from_cost + e_from_xt  # (B, N, N, H)


    time_emb = self.time_embed(timestep_embedding(timesteps.float(), self.hidden_dim))  # (B, time_embed_dim)


    graph_structure = torch.ones(B, N, N, device=cost_matrix.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e


      x, e = layer(x, e, graph_structure, mode="direct", sparse=False)


      time_emb_projected = time_layer(time_emb)  # (B, H)
      if not self.node_feature_only:
        e = e + time_emb_projected[:, None, None, :]  # (B,N,N,H) + (B,1,1,H)
      else:
        x = x + time_emb_projected[:, None, :]  # (B,N,H) + (B,1,H)


      x = x_in + x
      e = e_in + out_layer(e)


    logits = self.out(e.permute(0, 3, 1, 2))
    return logits

  def sparse_forward(self, cost_matrix_dense, timesteps, edge_index, xt):


    B, N, _ = cost_matrix_dense.shape
    outgoing_costs = cost_matrix_dense
    incoming_costs = cost_matrix_dense.transpose(1, 2)
    node_features_raw = torch.cat([outgoing_costs, incoming_costs], dim=-1)
    x = self.node_feature_embedder(node_features_raw.view(B * N, -1))

    if B > 1:
      batch_idx = torch.arange(B, device=edge_index.device).repeat_interleave(edge_index.shape[1] // B)
      edge_costs = cost_matrix_dense[batch_idx, edge_index[0], edge_index[1]]
    else:
      edge_costs = cost_matrix_dense[0, edge_index[0], edge_index[1]]

    e_from_cost = self.edge_embed(self.edge_cost_embed(edge_costs.view(1, 1, -1, 1)).squeeze(0).squeeze(0))
    e_from_xt = self.edge_embed(self.edge_cost_embed(xt.float().view(1, 1, -1, 1)).squeeze(0).squeeze(0))
    e = e_from_cost + e_from_xt

    time_emb = self.time_embed(timestep_embedding(timesteps.float(), self.hidden_dim))

    x, e = self.sparse_encoding(x, e, edge_index.long(), time_emb)


    logits = self.per_layer_out[-1](e)

    return logits

  def forward(self, cost_matrix, timesteps, xt, edge_index=None):

    if self.sparse:
      if edge_index is None:
        raise ValueError("edge_index must be provided for sparse mode.")

      return self.sparse_forward(cost_matrix, timesteps, edge_index, xt)
    else:
      return self.dense_forward(cost_matrix, timesteps, xt)

  def sparse_encoding(self, x, e, edge_index, time_emb):

    adj_matrix_sparse = SparseTensor(
      row=edge_index[0], col=edge_index[1],
      value=torch.ones_like(edge_index[0].float()),
      sparse_sizes=(x.shape[0], x.shape[0])
    ).to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e
      x, e = layer(x, e, adj_matrix_sparse, mode="direct", edge_index=edge_index, sparse=True)
      if not self.node_feature_only:
        e = e + time_layer(time_emb)
      else:
        x = x + time_layer(time_emb)
      x = x_in + x
      e = e_in + out_layer(e)
    return x, e
