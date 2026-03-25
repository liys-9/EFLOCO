import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
import torch.nn.functional as F


class UnionFind:
  def __init__(self, n):

    self.parent = list(range(n))

    self.rank = [0] * n

  def find(self, i):

    if self.parent[i] == i:
      return i
    self.parent[i] = self.find(self.parent[i])
    return self.parent[i]

  def union(self, i, j):

    root_i = self.find(i)
    root_j = self.find(j)
    if root_i != root_j:

      if self.rank[root_i] > self.rank[root_j]:
        self.parent[root_j] = root_i
      else:
        self.parent[root_i] = root_j
        if self.rank[root_i] == self.rank[root_j]:
          self.rank[root_j] += 1
      return True
    return False

def calculate_tour_distance(tours, dist_matrix):

  tours = np.array(tours)
  if tours.ndim == 1:
    tours = tours.reshape(1, -1)

  costs = []
  for tour in tours:
    cost = 0
    for i in range(len(tour) - 1):
      cost += dist_matrix[tour[i], tour[i + 1]]
    costs.append(cost)
  return np.array(torch.tensor(costs).cpu())
def build_atsp_tour_heuristic(likelihood_matrix, cost_matrix):

  num_nodes = likelihood_matrix.shape[0]

  edges = []
  for i in range(num_nodes):
    for j in range(num_nodes):
      if i == j:
        continue


      likelihood = likelihood_matrix[i, j]
      cost = cost_matrix[i, j]
      # score = likelihood * 10000
      score = likelihood / (cost + 1e-9)
      edges.append((score, i, j)) 


  edges.sort(key=lambda x: x[0], reverse=True)

  tour_edges = []
  uf = UnionFind(num_nodes)


  out_degree = np.zeros(num_nodes, dtype=int)
  in_degree = np.zeros(num_nodes, dtype=int)

  for score, u, v in edges:

    if len(tour_edges) == num_nodes:
      break


    if out_degree[u] == 0 and in_degree[v] == 0 and uf.find(u) != uf.find(v):
      tour_edges.append((u, v))
      out_degree[u] += 1
      in_degree[v] += 1
      uf.union(u, v)


  if len(tour_edges) < num_nodes:

    is_node_in_tour = np.zeros(num_nodes, dtype=bool)
    for u, v in tour_edges:
      is_node_in_tour[u] = True
      is_node_in_tour[v] = True

    remaining_nodes = [i for i, in_tour in enumerate(is_node_in_tour) if not in_tour]
    for i in range(len(remaining_nodes)):
      u = remaining_nodes[i]
      v = remaining_nodes[(i + 1) % len(remaining_nodes)]
      if out_degree[u] == 0 and in_degree[v] == 0:
        tour_edges.append((u, v))
        out_degree[u] += 1
        in_degree[v] += 1

  succ_dict = {u: v for u, v in tour_edges}


  curr = 0
  tour = []
  visited = set()  

  start_nodes = list(range(num_nodes))
  while start_nodes and not tour:
    start_node = start_nodes.pop(0)
    curr = start_node
    tour = []
    visited.clear()

    while len(tour) < num_nodes:
      if curr in visited:
        break
      visited.add(curr)
      tour.append(curr)
      curr = succ_dict.get(curr)
      if curr is None: 
        break


  if len(tour) != num_nodes:
    tour = []
    remaining = set(range(num_nodes))
    curr = 0

    while remaining:
      tour.append(curr)
      remaining.remove(curr)
      if remaining:
 
        next_node = min(remaining, key=lambda x: cost_matrix[curr][x])
        curr = next_node

  return tour


def extract_tours_from_likelihood(likelihood_matrices, cost_matrices, parallel_sampling=1):


  assert likelihood_matrices.shape[0] == cost_matrices.shape[0] == parallel_sampling
  B, N, _ = likelihood_matrices.shape


  all_tours = []
  for b in range(B):
    likelihood_matrix = likelihood_matrices[b]
    cost_matrix = cost_matrices[b]


    best_tour = None
    best_cost = float('inf')

    for start_node in range(1):
      tour = [start_node]
      remaining = set(range(N)) - {start_node}

      while remaining:
        current = tour[-1]

        next_node = max(remaining, key=lambda x: likelihood_matrix[current, x])
        tour.append(next_node)
        remaining.remove(next_node)


      cost = 0
      for i in range(N - 1):
        cost += cost_matrix[tour[i], tour[i + 1]]
      cost += cost_matrix[tour[-1], tour[0]] 

      if cost < best_cost:
        best_cost = cost
        best_tour = tour

    if best_tour is not None:

      tour_array = np.array(best_tour)

      optimized_tour, _ = batched_atsp_node_relocation_torch(
        cost_matrices=np.expand_dims(cost_matrix.cpu().numpy(), 0),
        initial_tours=np.expand_dims(tour_array, 0),
        max_iterations=1,
        device="cpu"
      )

      best_tour = optimized_tour[0].tolist()

    all_tours.append(best_tour)

  return all_tours


import torch
import numpy as np


#

def batched_atsp_node_relocation_torch(cost_matrices, initial_tours, max_iterations=100, device="cpu"):


  B, N = initial_tours.shape


  if isinstance(cost_matrices, torch.Tensor):
    cost_matrices = cost_matrices.cpu().numpy()
  if isinstance(initial_tours, torch.Tensor):
    initial_tours = initial_tours.cpu().numpy()

  with torch.inference_mode():
   
    costs_gpu = torch.from_numpy(cost_matrices).to(device)
    tours_gpu = torch.from_numpy(initial_tours).to(device)


    batch_idx = torch.arange(B, device=device)[:, None]


    improvements = torch.zeros(B, device=device)
    best_tours = tours_gpu.clone()
    best_costs = torch.zeros(B, device=device)

    for b in range(B):
      tour = tours_gpu[b]
      cost = 0
      for i in range(N - 1):
        cost += costs_gpu[b, tour[i], tour[i + 1]]
      cost += costs_gpu[b, tour[-1], tour[0]]  
      best_costs[b] = cost

    for iteration in range(max_iterations):

      tours_prev = torch.roll(tours_gpu, 1, dims=1)
      tours_next = torch.roll(tours_gpu, -1, dims=1)


      node_i = tours_gpu.view(B, N, 1)  # (B, N, 1)
      node_i_prev = tours_prev.view(B, N, 1)  # (B, N, 1)
      node_i_next = tours_next.view(B, N, 1)  # (B, N, 1)

      node_j = tours_gpu.view(B, 1, N)  # (B, 1, N)
      node_j_next = tours_next.view(B, 1, N)  # (B, 1, N)


      cost_i_prev_to_i = costs_gpu[batch_idx, node_i_prev, node_i]
      cost_i_to_i_next = costs_gpu[batch_idx, node_i, node_i_next]
      cost_j_to_j_next = costs_gpu[batch_idx, node_j, node_j_next]

      cost_i_prev_to_i_next = costs_gpu[batch_idx, node_i_prev, node_i_next]
      cost_j_to_i = costs_gpu[batch_idx, node_j, node_i]
      cost_i_to_j_next = costs_gpu[batch_idx, node_i, node_j_next]


      cost_removed = cost_i_prev_to_i + cost_i_to_i_next + cost_j_to_j_next
      cost_added = cost_i_prev_to_i_next + cost_j_to_i + cost_i_to_j_next
      change_matrix = cost_added - cost_removed


      mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
      i_pos_indices = torch.arange(N, device=device)[None, :, None]
      j_pos_indices = torch.arange(N, device=device)[None, None, :]
      mask.scatter_(2, i_pos_indices, True)  
      mask.scatter_(2, torch.roll(i_pos_indices, -1, dims=1), True)  

      change_matrix[mask] = float('inf')


      flat_change = change_matrix.view(B, -1)
      min_change_val, flat_best_move_idx = torch.min(flat_change, dim=1)


      if torch.all(min_change_val >= -1e-9):
        break


      best_i_pos = torch.div(flat_best_move_idx, N, rounding_mode='floor')
      best_j_pos = flat_best_move_idx % N


      new_tours_list = []
      for b in range(B):
        if min_change_val[b] < -1e-9:
          tour_list = tours_gpu[b].tolist()
          i_pos = best_i_pos[b].item()
          j_pos = best_j_pos[b].item()


          node_to_move = tour_list.pop(i_pos)


          if i_pos < j_pos:
            tour_list.insert(j_pos, node_to_move)
          else:
            tour_list.insert(j_pos + 1, node_to_move)

          new_tours_list.append(torch.tensor(tour_list, device=device))


          improvements[b] += min_change_val[b].item()
        else:
          new_tours_list.append(tours_gpu[b])

      tours_gpu = torch.stack(new_tours_list)


      for b in range(B):
        tour = tours_gpu[b]
        cost = 0
        for i in range(N - 1):
          cost += costs_gpu[b, tour[i], tour[i + 1]]
        cost += costs_gpu[b, tour[-1], tour[0]]

        if cost < best_costs[b]:
          best_costs[b] = cost
          best_tours[b] = tour.clone()

    return best_tours.cpu().numpy(), iteration + 1

def extract_tours_stochastic(likelihood_matrices, cost_matrices, num_samples=128, device="cpu"):
 
  assert likelihood_matrices.shape[0] == cost_matrices.shape[0]
  B, N, _ = likelihood_matrices.shape
  final_best_tours = []

  for b in range(B):
    likelihood = likelihood_matrices[b]
    costs = cost_matrices[b]

    initial_tours = []
    for _ in range(num_samples):
      start_node = np.random.randint(N)
      tour = [start_node]
      remaining = set(range(N)) - {start_node}

      while remaining:
        current_node = tour[-1]


        nodes_to_sample = list(remaining)
        weights = [likelihood[current_node, next_node] for next_node in nodes_to_sample]


        sum_weights = sum(weights)
        if sum_weights > 0:
          weights = [w / sum_weights for w in weights]
        else:  
          weights = [1.0 / len(nodes_to_sample)] * len(nodes_to_sample)

        next_node = np.random.choice(nodes_to_sample, p=weights)

        tour.append(next_node)
        remaining.remove(next_node)


      initial_tours.append(tour + [tour[0]])


    tours_array = np.array(initial_tours, dtype=np.int64)


    optimized_tours, _ = batched_atsp_two_opt_torch_guaranteed(
      dist_matrix=costs,
      tour=tours_array,
      max_iterations=1000, 
      device=device
    )

  
    tour_costs = calculate_tour_distance(optimized_tours[:, :-1], costs)
    best_tour_index = np.argmin(tour_costs)
    best_tour = optimized_tours[best_tour_index][:-1]  

    final_best_tours.append(best_tour.tolist())

  return final_best_tours


def batched_atsp_two_opt_torch_guaranteed(dist_matrix, tour, max_iterations=1000, device="cpu"):
  """
  Performs a batched 2-opt optimization for the Asymmetric TSP using PyTorch,
  with a guaranteed improvement on each swap.

  This function calculates the exact change in tour length for an ATSP 2-opt move
  by accounting for both the endpoint edge changes and the cost change of the
  reversed sub-path. This guarantees that every swap reduces the tour length.
  It uses a vectorized prefix-sum technique for efficient computation.

  Args:
    dist_matrix (np.ndarray): A square matrix of shape (num_nodes, num_nodes)
                              representing the distances between nodes. Can be
                              asymmetric.
    tour (np.ndarray): A batch of tours, with shape (batch_size, num_nodes + 1).
                       Each tour is a sequence of node indices, starting and
                       ending at the same node.
    max_iterations (int): The maximum number of iterations to perform.
    device (str): The computing device to use ('cpu' or 'cuda').

  Returns:
    tuple: A tuple containing:
      - np.ndarray: The optimized batch of tours.
      - int: The number of iterations performed.
  """
  iterator = 0
  tour = tour.copy()

  # Ensure distance matrix is a 2D tensor
  if dist_matrix.ndim == 3:
    dist_matrix = dist_matrix.squeeze(0)
  num_nodes = dist_matrix.shape[0]

  with torch.inference_mode():
    # Move data to the specified device
    # cuda_dist_matrix = torch.from_numpy(dist_matrix).float().to(device)
    # cuda_tour = torch.from_numpy(tour).long().to(device)
    cuda_dist_matrix = dist_matrix.float().to(device)
    cuda_tour = torch.tensor(tour).long().to(device)
    batch_size = cuda_tour.shape[0]

    min_change = -1.0
    while min_change < 0.0:
      # Get node indices for each position in the tours (excluding the last node)
      tour_i = cuda_tour[:, :-1]
      tour_i_plus_1 = cuda_tour[:, 1:]

      # --- 1. Calculate Endpoint Change (same as the heuristic) ---
      dist_current_edges = cuda_dist_matrix[tour_i, tour_i_plus_1]

      tour_i_b = tour_i.unsqueeze(2)
      tour_j_b = tour_i.unsqueeze(1)
      dist_new_ij = cuda_dist_matrix[tour_i_b, tour_j_b]

      tour_i_plus_1_b = tour_i_plus_1.unsqueeze(2)
      tour_j_plus_1_b = tour_i_plus_1.unsqueeze(1)
      dist_new_i1_j1 = cuda_dist_matrix[tour_i_plus_1_b, tour_j_plus_1_b]

      endpoint_change = (dist_new_ij + dist_new_i1_j1 -
                         dist_current_edges.unsqueeze(2) - dist_current_edges.unsqueeze(1))

      # --- 2. Calculate Internal Path Cost Change (The Correction) ---
      # This is the crucial part that guarantees improvement for ATSP.

      # Get costs of reversed edges: dist(tour[k+1], tour[k])
      d_backward = cuda_dist_matrix[tour_i_plus_1, tour_i]

      # Calculate prefix sums for both forward and backward path segments
      S_forward = torch.cumsum(dist_current_edges, dim=1)
      S_backward = torch.cumsum(d_backward, dim=1)

      # The change in cost from reversing the internal path from i+1 to j is:
      # Cost(j -> j-1 -> ... -> i+1) - Cost(i+1 -> i+2 -> ... -> j)
      # This can be calculated efficiently as:
      # (S_backward[j-1] - S_backward[i]) - (S_forward[j-1] - S_forward[i])
      # which simplifies to: (S_backward - S_forward)[j-1] - (S_backward - S_forward)[i]

      cost_diff = S_backward - S_forward

      # To vectorize the calculation M[i,j] = cost_diff[j-1] - cost_diff[i],
      # we pad cost_diff and use broadcasting.
      cost_diff_padded = F.pad(cost_diff, (1, 0))  # Pad left with 0

      term_j_minus_1 = cost_diff_padded[:, :-1].unsqueeze(1)  # For cost_diff[j-1]
      term_i = cost_diff_padded[:, 1:].unsqueeze(2)  # For cost_diff[i]

      # internal_path_change[b, i, j] is the exact cost change of reversing the subpath
      internal_path_change = term_j_minus_1 - term_i

      # --- 3. Calculate Total Accurate Change ---
      total_change = endpoint_change + internal_path_change

      # --- 4. Find Best Move and Apply ---
      # Mask out invalid swaps (j <= i+1) and non-improving moves
      valid_change = torch.triu(total_change, diagonal=2)
      valid_change[valid_change >= -1e-6] = float('inf')  # Use a small tolerance

      # Find the best swap for each tour in the batch
      min_change_per_batch, flatten_argmin_index = torch.min(
        valid_change.reshape(batch_size, -1), dim=-1
      )
      min_change = torch.min(min_change_per_batch)

      if min_change < -1e-6:
        # Decode the 1D index back to 2D indices (min_i, min_j) for each tour
        min_i = torch.div(flatten_argmin_index, num_nodes, rounding_mode='floor')
        min_j = torch.remainder(flatten_argmin_index, num_nodes)

        # Apply the 2-opt swap for each tour that has a valid improvement
        for b in range(batch_size):
          if min_change_per_batch[b] < -1e-6:
            i, j = min_i[b], min_j[b]
            cuda_tour[b, i + 1:j + 1] = torch.flip(cuda_tour[b, i + 1:j + 1], dims=(0,))
        iterator += 1
      else:
        break  # No more improvements found

      if iterator >= max_iterations:
        break

    # Copy the optimized tours back to the CPU
    tour = cuda_tour.cpu().numpy()

  return tour, iterator