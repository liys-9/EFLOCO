"""Lightning module for training the ATSP model."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info


from co_datasets.tsp_graph_atsp_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours,batched_two_opt_distance_matrix
from utils.discrete_schedulers import PolynomialConvexScheduler,MixtureDiscreteProbPath
from discrete_solver import MixtureDiscreteEulerSolver

class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    self.bad_model = None

    self.train_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.test_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.validation_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.T = 1.0  # Maximum time for flow matching (usually 1)
    self.time_steps = 100  # Number of time steps for probability estimation
    self.prob_update_freq = 10  # Update probabilities every this many steps
    self.t_all = torch.linspace(0.001, 1.0 - 1e-3, self.time_steps)
    self.t_probabilities = torch.ones_like(self.t_all) / self.time_steps  # Initialize uniform
    self.delta_t = 0.005  # Neighborhood size for sampling


  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)




  def compute_change_metric(self, points, xt, t, edge_index):
    with torch.no_grad():
      pred_vf = self.forward(
        points.float().to(points.device),
        xt.float().to(points.device),
        t.float().to(points.device),
        edge_index,
      )

        # Option 1: Magnitude of the vector field
    change_metric = torch.norm(pred_vf, dim=-1)
    return change_metric.mean()

  def generate_time_probabilities(self, batch):
      """
      Generates a probability distribution over 't' based on estimated change.

      Args:
          batch: The training batch of data.
      """
      self.model.eval()
      device = next(self.model.parameters()).device

      all_change = []
      for t in self.t_all.to(device):
          # Prepare batch data
          if not self.sparse:
              _, points, adj_matrix, _ = batch
          else:
              _, graph_data, point_indicator, edge_indicator, _ = batch
              points = graph_data.x

          batch_t = t.repeat(points.shape[0])
          # Create dummy xt (or use a meaningful input if available)
          # xt = torch.randn_like(points).to(device)
          xt = torch.randn_like(adj_matrix.float())
          xt = (xt > 0).long()
          all_change.append(
              self.compute_change_metric(
                  points.float().to(device),
                  xt.float().to(device),
                  batch_t,
                  None  # edge_index.to(device) if edge_index is not None else None
              ).cpu()
          )

      all_change = torch.stack(all_change)
      probabilities = F.softmax(all_change, dim=0)
      self.model.train()
      self.t_probabilities = probabilities.cpu()

  def generate_time_probabilities_optimal_grid_blended(self, batch, optimal_t_grid, blend_factor=0.5):
    """
    Generates a probability distribution over 't' based on estimated change,
    using pre-defined non-uniform time points (optimal_t_grid).  Blends
    probabilities from left and right endpoints of intervals.

    Args:
        batch: The training batch of data.
        optimal_t_grid: Pre-defined non-uniform time points.
        blend_factor: Weighting factor for blending probabilities (0.0 to 1.0).
    """
    self.model.eval()
    device = next(self.model.parameters()).device

    self.t_probabilities = torch.zeros(len(optimal_t_grid) - 1, device=device)  # Initialize probabilities

    for i in range(len(optimal_t_grid) - 1):
      t_left = optimal_t_grid[i]
      t_right = optimal_t_grid[i + 1]

      # 1. Create a set of t values within the interval
      num_interval_points = 5  # Reduced points for efficiency
      interval_t_all = torch.linspace(t_left.item(), t_right.item(), num_interval_points, device=device)

      # 2. Calculate change metric for these points
      interval_change = []
      if not self.sparse:
        _, points, adj_matrix, _ = batch
      else:
        _, graph_data, point_indicator, edge_indicator, _ = batch
        points = graph_data.x
        edge_index = graph_data.edge_index

      for t in interval_t_all:
        batch_t = t.repeat(points.shape[0])
        xt = torch.randn_like(points).to(device)  # Or your input
        interval_change.append(
          self.compute_change_metric(
            points.float().to(device),
            xt.float().to(device),
            batch_t,
            None  # edge_index.to(device) if edge_index is not None else None
          ).cpu()  # Shape: [batch]
        )
      if interval_change:  # Check if the list is not empty
          interval_change = torch.stack(interval_change)  # Shape: [num_interval_points, batch_size]
          interval_change = interval_change.mean(dim=1)  # Shape: [num_interval_points]
      else:
          # Handle the case where interval_change is empty (e.g., if num_interval_points is 0)
          interval_change = torch.zeros(num_interval_points, device=device)  # Or some appropriate default
      # interval_change = torch.stack(interval_change)  # Shape: [num_interval_points, batch]
      # # Average over the batch dimension to get a single change value for each
      # # time point within the interval.
      # interval_change = interval_change.mean(dim=1)  # Shape: [num_interval_points]
      interval_probabilities = F.softmax(interval_change, dim=0)  # Shape: [num_interval_points]

      # 3. Blend probabilities for left and right endpoints
      prob_left = interval_probabilities[0] if len(interval_probabilities) > 0 else torch.tensor(0.0, device=device)
      prob_right = interval_probabilities[-1] if len(interval_probabilities) > 0 else torch.tensor(0.0, device=device)

      blended_prob = blend_factor * prob_left + (1 - blend_factor) * prob_right

      self.t_probabilities[i] = blended_prob

    # 4. Normalize probabilities
    self.t_probabilities = F.softmax(self.t_probabilities, dim=0).cpu()
    self.model.train()
  def sample_t(self, batch_size, device):
      """
      Samples t values from the generated distribution.

      Args:
          batch_size: The batch size.
          device:  The device to use (CPU or GPU).

      Returns:
          A tensor of sampled t values.
      """
      indices = torch.multinomial(self.t_probabilities, num_samples=batch_size, replacement=True)
      return self.t_all[indices].to(device)

  def sample_t_continuous_approximation(self, batch_size, device, neighborhood_size=0.01):
      """
      Approximates continuous t sampling by sampling uniformly within a
      neighborhood around the discrete t_all values.

      Args:
          batch_size: The batch size.
          device: The device to use (CPU or GPU).
          neighborhood_size: The size of the neighborhood to sample within.

      Returns:
          A tensor of sampled t values (approximating continuous).
      """
      indices = torch.multinomial(self.t_probabilities, num_samples=batch_size, replacement=True)
      t_base = self.t_all[indices].to(device)  # The discrete t values

      # Sample offsets from a uniform distribution
      t_offsets = (torch.rand_like(t_base) * 2 - 1) * neighborhood_size
      t_sampled = torch.clamp(t_base + t_offsets, 0.001, 1.0)  # Clamp to valid range

      return t_sampled

  def threshold_based_f(self, xr, r, segment_ends_expand, x0_pred_r, x_at_segment_ends,threshold):
    if (threshold, int) and threshold == 0:
      return x_at_segment_ends
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    less_than_threshold = r < threshold
    from discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver
    solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                        vocabulary_size=2, sparse=self.sparse)

    # print('less_than_threshold',less_than_threshold.shape)
    x = solver.jump_state_to(xr, r, segment_ends_expand, x0_pred_r)
    # print('x',x.shape)
    less_than_threshold = less_than_threshold.view(-1, 1, 1) #no spr
    # less_than_threshold = less_than_threshold.view(-1, 1)
    res = (
        less_than_threshold * solver.jump_state_to(xr, r, segment_ends_expand, x0_pred_r)
        + (~less_than_threshold) * x_at_segment_ends
    )
    return res

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, dist_matrix, adj_matrix, _ = batch
      # print(adj_matrix)
      # t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
    else:
      _, graph_data, point_indicator, edge_indicator, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    use_step_optim = self.current_epoch >= 10  

    if use_step_optim:
        if not hasattr(self, 't_grid') or self.t_grid is None:
            from utils.step_optim import StepOptim
            from atsp_discrete_solver_Kinetic_optimed import get_or_load_t_grid
            scheduler = PolynomialConvexScheduler(n=2.0)
            self.path = MixtureDiscreteProbPath(scheduler=scheduler)
            self.t_grid = get_or_load_t_grid(
                scheduler=scheduler,
                nfe=20, 
                eps=1 - 1e-3,
                initType='unif_t',
                device=adj_matrix.device
            )

        
        def sample_time_uniform_interval(t_grid, batch_size, device):
            indices = torch.randint(0, len(t_grid) - 1, (batch_size,), device=device)
            t_left = t_grid[indices]
            t_right = t_grid[indices + 1]
            return t_left + torch.rand_like(t_left) * (t_right - t_left)

        t = sample_time_uniform_interval(self.t_grid, adj_matrix.shape[0], adj_matrix.device)

        seg_indices = torch.searchsorted(self.t_grid, t, side="left").clamp(
          min=1)  # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = self.t_grid[seg_indices]

    else:

        time_epsilon = 1e-3
        t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1.0 - time_epsilon)
        segments = torch.linspace(0, 1, 20, device=adj_matrix.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(
          min=1)  # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = segments[seg_indices]



    
    max_eps = segment_ends - t - 1e-6
    # con_delta = 1e-3
    con_delta = torch.full_like(max_eps, 1e-3)
    min_eps = torch.zeros_like(max_eps)
    con_delta = torch.clamp(con_delta, min=min_eps, max=max_eps)
    r = torch.clamp(t + con_delta, max=1.0)

    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    time_epsilon = 1e-3

    adj_matrix = adj_matrix.long()

    z0 = torch.randn_like(adj_matrix.float())

    z0 = (z0 > 0).long()

    path_sample = path.sample(t=t, x_0=z0, x_1=adj_matrix)
    xt = path_sample.x_t.float()

    path_sample_r = path.sample(t=r, x_0=z0, x_1=adj_matrix)
    xr = path_sample_r.x_t.float()

    path_sample_seg = path.sample(t=segment_ends, x_0=z0, x_1=adj_matrix)
    x_at_segment_ends = path_sample_seg.x_t.float()

    if self.sparse:
      # t = torch.from_numpy(t).float()
      print(t,t.shape)
      t_ = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      r_ = r.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      segment_ends_ = segment_ends.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      xt = xt.reshape(-1)
      xr = xr.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)

    rng_state = torch.cuda.get_rng_state()

    x0_pred = self.forward(
        dist_matrix.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    torch.cuda.set_rng_state(rng_state) # Shared Dropout Mask
    with torch.no_grad():

      x0_pred_r = self.forward(
        dist_matrix.float().to(adj_matrix.device),
        xr.float().to(adj_matrix.device),
        r.float().to(adj_matrix.device),
        edge_index,
      )
      x0_pred_r = torch.nan_to_num(x0_pred_r)

    from atsp_discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver
    solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                        vocabulary_size=2, sparse= self.sparse)


    ft = solver.jump_state_to(xt, t, segment_ends, x0_pred)
    threshold = 0.95

    fr = self.threshold_based_f(xr, r, segment_ends, x0_pred_r, x_at_segment_ends, threshold)

    ft = F.one_hot(ft.to(torch.int64), num_classes=2).float()
    fr = F.one_hot(fr.to(torch.int64), num_classes=2).float()

    loss_mse = F.mse_loss(ft, fr)
    loss_func = nn.CrossEntropyLoss().to(device=adj_matrix.device)

    loss = loss_func(x0_pred, adj_matrix.long())

    total_loss = loss + (m * loss_mse)

    self.log("train/loss", total_loss, prog_bar=True)
    self.log("mse", loss_mse, prog_bar=True)
    self.log("cross", loss, prog_bar=True)


    return loss


  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)




  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, dist_matrix, adj_matrix, gt_tour = batch
      # print('p',points.shape)
      # np_points = points.cpu().numpy()
      # np_gt_tour = gt_tour.cpu().numpy()
      # no batch
      np_dist_matrix = dist_matrix.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
      # print(adj_matrix)
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    stacked_tours = []
    ns, merge_iterations = 0, 0

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        dist_matrix = dist_matrix.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)



      if self.diffusion_type == 'gaussian':
          with torch.no_grad():

              points = points.to(device)
              batch_size = adj_matrix.shape[0]
              shape = (batch_size, 50, 50)

              # x = z0_in.to(model.device)
              # print('z',x)



              # model_fn = mutils.get_model_fn(model, train=False)

              ### Uniform
              NFE = 32
              dt = 1. / NFE
              eps = 1e-3  # default: 1e-3
              for i in range(NFE):
                  num_t = i / NFE * (1 - eps) + eps
                  # print(num_t)
                  vec_t = torch.ones(shape[0], device=device) * num_t

                  batch_size, node_size, _ = adj_matrix.shape


                  drift = self.forward(
                      points,
                      xt,
                      vec_t * 999,
                      # None,
                      None,
                  )
                  drift = drift.squeeze(1)

                  xt = xt + drift * dt



      else:

          from atsp_discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver
         
          scheduler = PolynomialConvexScheduler(n)
          path = MixtureDiscreteProbPath(scheduler=scheduler)
         

          solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                              vocabulary_size=2,sparse= self.sparse)
         
          sol = solver.sample(
              x_init=xt,
              step_size=step_size,
              verbose=False,
              return_intermediates=True,
              eval_batch=batch,
              use_step_optim=True,  
              nfe=2,  
              step_optim_init_type='unif', 
              time_grid=torch.tensor([0.0, 1 - 1e-3]), 

          )
          xt = sol

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)



      from utils.atsp_utils import extract_tours_from_likelihood,batched_atsp_node_relocation_torch,extract_tours_stochastic


      tours = extract_tours_from_likelihood(
          adj_mat, dist_matrix,
          # np_edge_index,
          # sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )



      def calculate_tour_length_np(cost_matrix, tour):
        if isinstance(tour, torch.Tensor):
          tour = tour.squeeze(0).cpu().numpy()
        if isinstance(cost_matrix, torch.Tensor):
          cost_matrix = cost_matrix.squeeze(0).cpu().numpy()


        rolled_tour = np.roll(tour, -1)
        return cost_matrix[tour, rolled_tour].sum()



      improved_tour = iterated_local_search(tours, dist_matrix)
      # improved_tour, _ = batched_atsp_two_opt_torch_guaranteed(dist_matrix, tours, max_iterations=1000, device="cpu")
      print(tours,improved_tour)
      # stacked_tours.append(tours)
      stacked_tours.append(improved_tour)

      # print(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)
    print(solved_tours,solved_tours.shape)
    # tours = np.concatenate(tours, axis=0)

    initial_lengths = [calculate_tour_length_np(dist_matrix[0], np.array(solved_tours)[i]) for i in
                       range(solved_tours.shape[0])]
    print(initial_lengths)



  
    gt_cost = calculate_tour_length_np(dist_matrix, gt_tour)

   
    best_solved_cost = np.min(initial_lengths)



    gap = (best_solved_cost - gt_cost) / gt_cost * 100
 
    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": ns,
        # f"{split}/merge_iterations": merge_iterations,
        f"{split}/gap(%)": gap,
        # f"{split}/newgap(%)": imgap,
    }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):

    return self.test_step(batch, batch_idx, split='val')

 


