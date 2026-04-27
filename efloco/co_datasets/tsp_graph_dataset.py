"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch
from scipy.spatial import ConvexHull


from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class TSPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1

    return points, tour

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
#       sparse_factor = self.sparse_factor
#       kdt = KDTree(points, leaf_size=30, metric='euclidean')
#       dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

#       edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
#       edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

#       edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

#       tour_edges = np.zeros(points.shape[0], dtype=np.int64)
#       tour_edges[tour[:-1]] = tour[1:]
#       tour_edges = torch.from_numpy(tour_edges)
#       tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
#       tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
#       graph_data = GraphData(x=torch.from_numpy(points).float(),
#                              edge_index=edge_index,
#                              edge_attr=tour_edges)

#       point_indicator = np.array([points.shape[0]], dtype=np.int64)
#       edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
#       return (
#           torch.LongTensor(np.array([idx], dtype=np.int64)),
#           graph_data,
#           torch.from_numpy(point_indicator).long(),
#           torch.from_numpy(edge_indicator).long(),
#           torch.from_numpy(tour).long(),
#       )
      k = self.sparse_factor
      num_nodes = points.shape[0]
  
      # Step 2: Precompute convex hull and KNN information
      # Convex hull information
      hull = ConvexHull(points)
      hull_nodes = set(hull.vertices)
      hull_neighbor_map = {}
      for i in range(len(hull.vertices)):
        current_node = hull.vertices[i]
        prev_node = hull.vertices[i - 1]
        next_node = hull.vertices[(i + 1) % len(hull.vertices)]
        hull_neighbor_map[current_node] = {prev_node, next_node}
  
      # KNN information (use a few extra candidates for safety, e.g. k+2)
      kdt = KDTree(points, leaf_size=30, metric='euclidean')
      _, knn_idx = kdt.query(points, k=min(k + 2, num_nodes - 1))
  
      # Step 3: Select the best K neighbors independently for each node
      final_edge_list = []
      for i in range(num_nodes):
        node_i_neighbors = set()
  
        # Priority 1: add convex-hull neighbors
        if i in hull_nodes:
          node_i_neighbors.update(hull_neighbor_map[i])
  
        # Priority 2: add KNN neighbors until reaching K
        for neighbor in knn_idx[i]:
          if len(node_i_neighbors) >= k:
            break
          # Avoid self-loops and duplicates
          if neighbor != i:
            node_i_neighbors.add(neighbor)
  
        # Create edges from node i to its selected neighbors
        for neighbor in node_i_neighbors:
          final_edge_list.append((i, neighbor))
  
      # Create the final edge_index
      # Note: this is a directed graph; each node has out-degree K
      final_edge_index = torch.tensor(final_edge_list, dtype=torch.long).t().contiguous()
  
      # Step 4: compute 0/1 labels for the final edge list
      optimal_tour_edges = set()
      for i in range(len(tour) - 1):
        # For a directed graph, the edge direction matters
        optimal_tour_edges.add((tour[i], tour[i + 1]))
  
      final_tour_labels = []
      for i in range(final_edge_index.shape[1]):
        u, v = final_edge_index[0, i].item(), final_edge_index[1, i].item()
        # Check both directions since the TSP tour forms a cycle
        if (u, v) in optimal_tour_edges or (v, u) in optimal_tour_edges:
          final_tour_labels.append(1)
        else:
          final_tour_labels.append(0)
  
      final_edge_attr = torch.tensor(final_tour_labels, dtype=torch.float).reshape(-1, 1)
  
      # --- Create the GraphData object ---
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=final_edge_index,
                             edge_attr=final_edge_attr)
  
      point_indicator = np.array([num_nodes], dtype=np.int64)
      edge_indicator = np.array([graph_data.num_edges], dtype=np.int64)
  
      # --- Return values (same structure/semantics as before) ---
      return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        graph_data,
        torch.from_numpy(point_indicator).long(),
        torch.from_numpy(edge_indicator).long(),
        torch.from_numpy(tour).long(),
      )