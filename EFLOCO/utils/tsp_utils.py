import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from utils.cython_merge.cython_merge import merge_cython


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  # print(iterator)
  return tour, iterator

def batched_two_opt_distance_matrix(distance_matrix, tour, max_iterations=1000, device="cpu"):
  """
  distance_matrix: [N, N] or [B, N, N]  # 支持 batch 维或全共享距离矩阵
  tour: [B, N]  # 每个 batch 的初始路径
  """
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_tour = torch.from_numpy(tour).to(device='cpu')  # [B, N]
    cuda_tour = cuda_tour[:, :-1]
    print(cuda_tour.shape)
    # distance_matrix = distance_matrix.to(device='cpu')
    B, N = cuda_tour.shape
    cuda_tour = cuda_tour
    # N = 50
    print(cuda_tour.shape)

    print(distance_matrix.shape)
    # N = cuda_tour.shape

    if isinstance(distance_matrix, np.ndarray):
      distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32, device=device)
    if distance_matrix.dim() == 2:
      distance_matrix = distance_matrix.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

    assert distance_matrix.shape == (B, N, N), \
      f"Expected distance_matrix shape [B={B}, N={N}, N={N}], got {distance_matrix.shape}"

    min_change = -1.0
    while min_change < 0.0:
      # i and j are indices of current edges (i -> i+1), (j -> j+1)
      i_idx = cuda_tour[:, :-1]  # [B, N-1]
      j_idx = cuda_tour[:, 1:]  # [B, N-1]

      # Construct pairwise combinations (i, j) for the change matrix
      I = i_idx.unsqueeze(2).expand(B, N - 1, N - 1)  # [B, N-1, N-1]
      J = i_idx.unsqueeze(1).expand(B, N - 1, N - 1)
      I1 = j_idx.unsqueeze(2).expand(B, N - 1, N - 1)
      J1 = j_idx.unsqueeze(1).expand(B, N - 1, N - 1)

      # Batch index tensor
      batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(I)

      # Gather distances via index
      A_ij = distance_matrix[batch_idx, I, J]  # [B, N-1, N-1]
      A_i1j1 = distance_matrix[batch_idx, I1, J1]
      A_ii1 = distance_matrix[batch_idx, I, I1]
      A_jj1 = distance_matrix[batch_idx, J, J1]

      # Compute change in cost if (i, i+1) and (j, j+1) are replaced
      change = A_ij + A_i1j1 - A_ii1 - A_jj1  # [B, N-1, N-1]

      # Only consider i < j - 1 to ensure valid, non-overlapping swaps
      triu_mask = torch.triu(torch.ones(N - 1, N - 1, device=device), diagonal=2).bool()
      valid_change = change.masked_fill(~triu_mask, float('inf'))

      # If no valid change remains
      if torch.isinf(valid_change).all():
        break

      # Find best (i, j) for each batch
      flat_change = valid_change.view(B, -1)  # [B, (N-1)*(N-1)]
      flat_argmin = flat_change.argmin(dim=1)  # [B]
      min_change = flat_change.gather(1, flat_argmin.unsqueeze(1)).min().item()

      # Get corresponding i, j indices
      min_i = flat_argmin // (N - 1)
      min_j = flat_argmin % (N - 1)

      # Apply flip if improvement is found
      if min_change < -1e-6:
        for b in range(B):
          if min_j[b] > min_i[b]:
            i = min_i[b].item()
            j = min_j[b].item()
            cuda_tour[b, i + 1:j + 1] = torch.flip(cuda_tour[b, i + 1:j + 1], dims=[0])
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break

    return cuda_tour.cpu().numpy(), iterator


def numpy_merge(points, adj_mat):
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)
  merge_iterations = 0
  for edge in (-adj_mat / dists).flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
    real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
        ).toarray() for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          cython_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations





def batched_two_opt_torch_new(points_batch, tours_batch_with_sampling, max_iterations=1000, device="cpu"):
  """
  一个能同时处理批次(Batch)和采样(Sampling)维度的、最终修正版的 2-Opt 函数。

  Args:
      points_batch (np.ndarray):
          点坐标批次。形状应为 (B, N, 2)，其中 B 是批次大小, N 是节点数。

      tours_batch_with_sampling (np.ndarray):
          路径批次，包含每个问题的多个候选解。
          形状应为 (B, S, N + 1)，其中 S 是每个问题的候选解数量。

      max_iterations (int):
          最大迭代次数。

      device (str):
          计算设备，例如 "cpu" 或 "cuda:0"。

  Returns:
      np.ndarray: 优化后的路径批次，形状恢复为 (B, S, N + 1)。
      int: 实际迭代次数。
  """
  # --- 1. 验证和准备数据 ---
  if points_batch.ndim != 3 or tours_batch_with_sampling.ndim != 3:
    raise ValueError(f"Incorrect input dimensions. Expected points (B,N,D) and tours (B,S,L), "
                     f"but got {points_batch.shape} and {tours_batch_with_sampling.shape}")

  batch_size_B, num_nodes_N, point_dims_D = points_batch.shape
  _, num_samples_S, tour_len_L = tours_batch_with_sampling.shape

  if points_batch.shape[0] != tours_batch_with_sampling.shape[0]:
    raise ValueError(
      f"Batch size mismatch between points ({batch_size_B}) and tours ({tours_batch_with_sampling.shape[0]})")

  # 将(B, S, L)的tours重塑为(B*S, L)，以便进行批处理
  tours_flat = tours_batch_with_sampling.reshape(batch_size_B * num_samples_S, tour_len_L)

  # 将(B, N, D)的points扩展为(B*S, N, D)，以匹配扁平化的tours
  # 每个问题的点坐标集需要重复S次
  points_expanded = np.repeat(points_batch, repeats=num_samples_S, axis=0)

  iterator = 0
  with torch.inference_mode():
    # 将准备好的 NumPy 数据转换为 PyTorch Tensors
    # 注意：现在所有张量的第一个维度都是 B*S
    cuda_points = torch.from_numpy(points_expanded).to(device, non_blocking=True)
    cuda_tours = torch.from_numpy(tours_flat).to(device, non_blocking=True)

    total_batch_size = cuda_points.shape[0]  # 这是 B * S

    min_change_val = torch.tensor(-1.0, device=device)
    while min_change_val < -1e-6:
      # --- 2. 核心计算逻辑 (与之前修正版类似，但现在作用于 B*S 的维度) ---
      batch_indices = torch.arange(total_batch_size, device=device).unsqueeze(1)
      tour_segments = cuda_tours[:, :-1]
      points_flat = cuda_points[batch_indices, tour_segments]

      points_i_bcast = points_flat.unsqueeze(2)
      points_j_bcast = points_flat.unsqueeze(1)

      points_i_plus_1_flat = cuda_points[batch_indices, cuda_tours[:, 1:]]

      dist_ij_sq = torch.sum((points_i_bcast - points_j_bcast) ** 2, dim=-1)
      dist_i1j1_sq = torch.sum((points_i_plus_1_flat.unsqueeze(2) - points_i_plus_1_flat.unsqueeze(1)) ** 2, dim=-1)
      dist_ii1_sq = torch.sum((points_flat - points_i_plus_1_flat) ** 2, dim=-1)

      change = (torch.sqrt(dist_ij_sq) + torch.sqrt(dist_i1j1_sq) -
                dist_ii1_sq.unsqueeze(2) - dist_ii1_sq.unsqueeze(1))

      mask = torch.ones_like(change, dtype=torch.bool)
      i_indices = torch.arange(num_nodes_N, device=device).unsqueeze(1)
      j_indices = torch.arange(num_nodes_N, device=device).unsqueeze(0)
      mask[:, (i_indices < j_indices - 1)] = False

      valid_change = change.clone()
      valid_change[mask] = float('inf')

      min_val_per_sample, flatten_argmin_index = torch.min(valid_change.view(total_batch_size, -1), dim=-1)
      min_change_val = torch.min(min_val_per_sample)

      if min_change_val < -1e-6:
        min_i = torch.div(flatten_argmin_index, num_nodes_N, rounding_mode='floor')
        min_j = torch.remainder(flatten_argmin_index, num_nodes_N)

        for i in range(total_batch_size):
          if min_val_per_sample[i] < -1e-6:
            start_node_idx, end_node_idx = min_i[i].item(), min_j[i].item()
            if start_node_idx > end_node_idx: start_node_idx, end_node_idx = end_node_idx, start_node_idx
            segment_to_flip = cuda_tours[i, start_node_idx + 1: end_node_idx + 1]
            cuda_tours[i, start_node_idx + 1: end_node_idx + 1] = torch.flip(segment_to_flip, dims=(0,))
        iterator += 1
      else:
        break
      if iterator >= max_iterations:
        break

  solved_tours_flat = cuda_tours.cpu().numpy()

  # --- 3. 将结果重塑回 (B, S, N+1) ---
  solved_tours_batched = solved_tours_flat.reshape(batch_size_B, num_samples_S, tour_len_L)

  return solved_tours_batched, iterator
#
#
def process_batch_of_candidates(adj_mats_all, points_all, edge_indices_all=None, sparse_graph=False,
                                parallel_sampling=1):
  """
  一个高级接口，能同时处理一个批次(B)的多个问题实例，以及每个实例的多个候选解(C)。

  Args:
      adj_mats_all (np.ndarray):
          所有候选邻接矩阵。形状应为 (B*C, N, N) 或 (B*C, E)。
          B=batch_size, C=每个实例的候选解数量。

      points_all (np.ndarray):
          所有问题实例的点坐标。形状为 (B, N, 2)。

      edge_indices_all (np.ndarray, optional):
          所有问题实例的边索引。形状 (B, 2, E)。

      sparse_graph (bool): 是否为稀疏模式。

      parallel_sampling (int): 控制并行进程数。

  Returns:
      np.ndarray: 所有路径，形状 (B, C, N + 1)。
      float: 平均合并迭代次数。
  """
  if points_all.ndim != 3:
    raise ValueError(f"Expected points_all to be batched (B, N, D), but got shape {points_all.shape}")

  batch_size_B = points_all.shape[0]
  num_total_adj_mats = adj_mats_all.shape[0]

  if num_total_adj_mats % batch_size_B != 0:
    raise ValueError(f"Total number of adjacency matrices ({num_total_adj_mats}) "
                     f"is not divisible by batch_size ({batch_size_B}).")

  candidates_per_instance_C = num_total_adj_mats // batch_size_B

  # --- 准备并行任务列表 ---
  # 每个任务是一个元组 (points, adj_mat, sparse_graph, edge_index)
  tasks = []
  for b_idx in range(batch_size_B):
    points_single = points_all[b_idx]
    for c_idx in range(candidates_per_instance_C):
      adj_mat_idx = b_idx * candidates_per_instance_C + c_idx
      adj_mat_single = adj_mats_all[adj_mat_idx]

      edge_index_single = edge_indices_all[b_idx] if sparse_graph and edge_indices_all is not None else None

      tasks.append((points_single, adj_mat_single, sparse_graph, edge_index_single))

  # --- 并行执行所有 B*C 个任务 ---
  effective_parallelism = min(len(tasks), parallel_sampling) if parallel_sampling > 1 else 1

  if len(tasks) > 1 and effective_parallelism > 1:
    with Pool(effective_parallelism) as p:
      results = p.starmap(_create_tour_from_adj, tasks)
  else:
    results = [_create_tour_from_adj(*task) for task in tasks]

  if not results:
    return np.array([]), 0.0

  # --- 收集并重塑结果 ---
  all_tours_flat, all_merge_iterations = zip(*results)

  # 将扁平化的 tour 列表重塑为 (B, C, N+1)
  num_nodes = points_all.shape[1]
  batched_tours = np.array(all_tours_flat, dtype=np.int64).reshape(
    batch_size_B, candidates_per_instance_C, num_nodes + 1
  )

  avg_merge_iterations = np.mean(all_merge_iterations)

  return batched_tours, avg_merge_iterations

def _create_tour_from_adj(points, adj_mat, sparse_graph, edge_index):
  """
  (内部辅助函数) 处理单个邻接矩阵，生成单个路径。
  这是并行池中的工作单元。
  """
  num_nodes = points.shape[0]

  # 1. 对称化 / 稀疏转密集
  if not sparse_graph:
    processed_adj_mat = adj_mat + adj_mat.T
  else:
    if edge_index is None:
      raise ValueError("edge_index is required for sparse graphs.")
    mat = scipy.sparse.coo_matrix(
      (adj_mat, (edge_index[0], edge_index[1])),
      shape=(num_nodes, num_nodes)
    ).toarray()
    processed_adj_mat = mat + mat.T

  # 2. 调用 Cython 合并
  real_adj_mat, merge_iterations = cython_merge(points, processed_adj_mat)

  # 3. 重建路径
  tour = [0]
  visited_nodes = {0}
  while len(tour) < num_nodes:
    last_node = tour[-1]
    neighbors = np.nonzero(real_adj_mat[last_node])[0]
    next_node = -1
    for neighbor in neighbors:
      if neighbor not in visited_nodes:
        next_node = neighbor
        break
    if next_node != -1:
      tour.append(next_node)
      visited_nodes.add(next_node)
    else:
      break

  if len(tour) == num_nodes:
    tour.append(tour[0])
  else:
    tour.extend([-1] * (num_nodes + 1 - len(tour)))

  return tour, merge_iterations

class TSPEvaluator(object):
  def __init__(self, points):
    self.dist_mat = scipy.spatial.distance_matrix(points, points)

  def evaluate(self, route):

    # 自动判断是否需要偏移：若最小值为 1，则视为从1开始编号
    if route.min() == 1:
      route -= 1
      print(route)

    total_cost = 0
    for i in range(len(route) - 1):
      total_cost += self.dist_mat[route[i], route[i + 1]]
    return total_cost





class BatchedTSPEvaluator:
  """
  一个完全矢量化的、支持批处理的TSP评估器。
  """

  def __init__(self, points_batch):
    """
    使用一批点坐标来初始化评估器。

    Args:
        points_batch (np.ndarray): 形状为 (B, N, D) 的点坐标批次。
                                   B=批次大小, N=节点数, D=坐标维度(通常为2)。
    """
    if points_batch.ndim != 3:
      raise ValueError(f"Expected points_batch to be 3D (B, N, D), but got shape {points_batch.shape}")

    # --- 批量计算距离矩阵 ---
    # 利用 NumPy 的广播机制，避免循环
    # points_batch (B, N, D) -> p1 (B, N, 1, D)
    p1 = np.expand_dims(points_batch, axis=2)
    # points_batch (B, N, D) -> p2 (B, 1, N, D)
    p2 = np.expand_dims(points_batch, axis=1)

    # p1 - p2 -> (B, N, N, D)
    # 对最后一个维度求和再开方，得到 (B, N, N) 的距离矩阵批次
    dist_sq = np.sum((p1 - p2) ** 2, axis=-1)
    self.dist_mats = np.sqrt(dist_sq)  # Shape: (B, N, N)

  def evaluate(self, routes_batch):
    """
    一次性评估一批路径的成本。

    Args:
        routes_batch (np.ndarray): 形状为 (B, L) 的路径批次。
                                    B=批次大小, L=路径长度(通常为N+1)。

    Returns:
        np.ndarray: 包含每条路径成本的一维数组，形状为 (B,)。
    """
    if routes_batch.ndim != 2:
      raise ValueError(f"Expected routes_batch to be 2D (B, L), but got shape {routes_batch.shape}")

    if routes_batch.shape[0] != self.dist_mats.shape[0]:
      raise ValueError(f"Batch size mismatch between routes ({routes_batch.shape[0]}) "
                       f"and the evaluator's points ({self.dist_mats.shape[0]})")

    # (可选) 检查并处理从1开始的编号，这里假设所有路径编号方式一致
    if routes_batch.min() == 1:
      routes_batch = routes_batch - 1

    # --- 矢量化成本计算 ---
    # 提取路径中所有边的起点和终点
    from_nodes = routes_batch[:, :-1]  # Shape: (B, N)
    to_nodes = routes_batch[:, 1:]  # Shape: (B, N)

    # 创建批次索引，用于高级索引
    batch_indices = np.arange(routes_batch.shape[0])[:, np.newaxis]  # Shape: (B, 1)

    # 使用高级索引一次性从距离矩阵中取出所有边的长度
    # self.dist_mats[batch_indices, from_nodes, to_nodes] 的结果形状为 (B, N)
    # 每一行包含了对应路径的所有边的长度
    edge_lengths = self.dist_mats[batch_indices, from_nodes, to_nodes]

    # 沿路径维度求和，得到每条路径的总成本
    total_costs = np.sum(edge_lengths, axis=1)  # Shape: (B,)

    return total_costs