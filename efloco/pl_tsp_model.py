"""Lightning module for the EFLOCO TSP model (legacy layout)."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info


from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
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
    self.t_grid = None


  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)

  # def on_epoch_start(self):
  #     # Get current epoch
  #     epoch = self.current_epoch  # Current training epoch
  #     return epoch



  def compute_change_metric(self, points, xt, t, edge_index):
    with torch.no_grad():
      pred_vf = self.forward(
        points.float().to(points.device),
        xt.float().to(points.device),
        t.float().to(points.device),
        edge_index,
      )
      # print(pred_vf.shape)

        # Option 1: Magnitude of the vector field
      change_metric = torch.norm(pred_vf, dim=1)  # Shape: [batch, 50, 50]

        # 2. Average over all node pairs (spatial dimensions)
        #    This gives a single value per graph in the batch, representing
        #    the overall "change" in the edge probabilities.
      change_metric = change_metric.mean(dim=[-1, -2])  # Shape: [batch]
      # print("chage:",change_metric,change_metric.shape)
    return change_metric

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
      mean_change_over_batch = all_change.mean(dim=1)  # Shape: [num_t_steps]
      probabilities = F.softmax(mean_change_over_batch, dim=0)
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
        xt = torch.randn_like(adj_matrix.float())
        xt = (xt > 0).long()
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
          # print(interval_change,interval_change.shape)
          interval_change = interval_change.mean(dim=1)  # Shape: [num_interval_points]
      else:
          # Handle the case where interval_change is empty (e.g., if num_interval_points is 0)
          interval_change = torch.zeros(num_interval_points, device=device)  # Or some appropriate default
      
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

  def sample_t_from_optimal_grid(self, batch_size, device, optimal_t_grid):
    """
    Samples t from intervals defined by optimal_t_grid, weighted by probabilities.
    Vectorized version for efficiency.
    """
    indices = torch.multinomial(self.t_probabilities, num_samples=batch_size, replacement=True).to(device)

    # Get the left and right boundaries of the intervals
    t_left = optimal_t_grid[indices]
    # Handle the last interval correctly using a combination of slicing and padding
    indices_right = torch.clamp(indices + 1, max=len(optimal_t_grid) - 1)
    t_right = optimal_t_grid[indices_right]

    # Sample uniformly within the intervals
    random_offsets = torch.rand(batch_size, device=device)
    t_sampled = t_left + random_offsets * (t_right - t_left)

    return t_sampled

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, adj_matrix, _ = batch
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

    use_step_optim = self.current_epoch >= 10  # Enable StepOptim after this epoch



    if use_step_optim:
        if not hasattr(self, 't_grid') or self.t_grid is None:
            from utils.step_optim import StepOptim
            from discrete_solver_Kinetic_optimed import get_or_load_t_grid
            scheduler = PolynomialConvexScheduler(n=2.0)
            self.path = MixtureDiscreteProbPath(scheduler=scheduler)
            self.t_grid = get_or_load_t_grid(
                scheduler=scheduler,
                nfe=20,  # Number of steps to simulate
                eps=1e-3,
                initType='unif_t',
                device=adj_matrix.device
            )

        # Uniformly sample t within intervals from the optimized grid
        def sample_time_uniform_interval(t_grid, batch_size, device):
            indices = torch.randint(0, len(t_grid) - 1, (batch_size,), device=device)
            t_left = t_grid[indices]
            t_right = t_grid[indices + 1]
            return t_left + torch.rand_like(t_left) * (t_right - t_left)
        
        t = sample_time_uniform_interval(self.t_grid, adj_matrix.shape[0], adj_matrix.device)

    else:
        # Early training: sample t from a uniform distribution
        time_epsilon = 1e-3
        t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1.0 - time_epsilon)

    # Do not use a time-series distribution here
    # if batch_idx % self.prob_update_freq == 0:
    #     self.generate_time_probabilities(batch)  # Update probabilities periodically
    #
    # # t = self.sample_t(points.shape[0], points.device)  # Sample t (discrete)
    # t = self.sample_t_continuous_approximation(
    #     points.shape[0], points.device, neighborhood_size=self.delta_t
    # )  # Sample t (approx. continuous)
    # # xt = self.diffusion.sample(adj_matrix_onehot, t)


#     if self.t_grid is not None and batch_idx % self.prob_update_freq == 0:
#         # self.generate_time_probabilities(batch)
#         self.generate_time_probabilities_optimal_grid_blended(batch, self.t_grid)
#         # print(self.generate_time_probabilities_optimal_grid_blended)

#     if self.t_grid is not None:
#         t = self.sample_t_from_optimal_grid(points.shape[0], points.device, self.t_grid)
        
#     else:
#         time_epsilon = 1e-3
#         t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1.0 - time_epsilon)


    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    time_epsilon = 1e-3
    # t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1.0 - time_epsilon)
    # t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1 - 2 * time_epsilon) + time_epsilon
    # from utils.t_dist import ExponentialPDF, sample_t
    # exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')
    # t = sample_t(exponential_distribution, adj_matrix.shape[0], 2).to(adj_matrix.device)
    z0 = torch.randn_like(adj_matrix)
    z0 = (z0 > 0).long()

    from utils.noise_matching import MatchScopeManager
    batch_size = adj_matrix.shape[0]
    # ms_size = batch_size
    # ms_mgr = MatchScopeManager(ms_size, h=50, w=50, device=adj_matrix.device)
    # if ms_size > 0:
    #   for i in range(batch_size):  # For each image, find the nearest noise in ns
    #     ms_mgr.assign_nearest(adj_matrix, z0, i)
    path_sample = path.sample(t=t, x_0=z0, x_1=adj_matrix)
    xt = path_sample.x_t.float()

    # In the original version, do not comment this out
    # xt = xt * 2 - 1
    # xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    if self.sparse:
      t = torch.from_numpy(t).float()
      t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      xt = xt.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
    # else:
      # t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
      # ts = self.diffusion.T + 1 -t
      # t = t / 999.0
      # eps = 1e-3

    # Denoise
    # print(adj_matrix.device)
    # print(next(self.model.parameters()).device)
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )
    # print('x0',x0_pred.shape,x0_pred)
    # weights = self.get_edge_weights(adj_matrix) * 0.1  # without weight decay

    # weights = self.get_edge_weights_decay_add(adj_matrix, self.current_epoch)
    # weights = self.get_flowmatching_style_weights(adj_matrix, t)
    # Compute loss
    # sample_weights = self.compute_sample_weights(adj_matrix, t)
    loss_func = nn.CrossEntropyLoss().to(device=adj_matrix.device)
    # loss_func = nn.CrossEntropyLoss(reduction='none')
    loss = loss_func(x0_pred, adj_matrix.long())

    # loss = (loss * sample_weights).mean()
    # from KLloss import MixturePathGeneralizedKL
    #
    # loss_func_kl = MixturePathGeneralizedKL(path=path)
    # loss = loss_func_kl(logits=x0_pred, x_1=adj_matrix, x_t=xt, t=t)

    # from KLloss import MixturePathGeneralizedKLKO
    #
    # loss_func_kl = MixturePathGeneralizedKLKO(path=path)
    # loss = loss_func_kl(logits=x0_pred, x_1=adj_matrix, x_t=xt, t=t)

    # # Kinetic-optimal loss
    # pt = path.prob(adj_matrix, t, adj_matrix)  # [B,N,N]
    # pt_p = path.prob(adj_matrix, t + time_epsilon, adj_matrix)
    # pt_m = path.prob(adj_matrix, t - time_epsilon, adj_matrix)
    #
    # # Finite-difference approximation of the time derivative ṗ_t
    # dp = (pt_p - pt_m) / (2 * time_epsilon)        # [B,N,N]
    #
    # # Kinetic-optimal flux: j* = [ p(z)*ṗ(x) - ṗ(z)*p(x) ]_+
    # # First broadcast to 4D
    # p_x = pt.unsqueeze(2)                # [B,N,1,N]
    # p_z = pt.unsqueeze(1)                # [B,1,N,N]
    # dp_x = dp.unsqueeze(2)               # [B,N,1,N]
    # dp_z = dp.unsqueeze(1)               # [B,1,N,N]
    #
    # num = p_z * dp_x - dp_z * p_x        # [B,N,N,N]
    # u_star = F.relu(num).sum(dim=2)      # merge the middle dimension -> [B,N,N]
    # # u_star = u_star * adj_matrix       # mask non-edges
    #
    # # Normalize to a jump distribution
    # target = u_star / (u_star.sum(dim=-1, keepdim=True) + 1e-8)  # [B,N,N]
    #
    # # Model-predicted logits: [B,C,N,N]
    # logits = x0_pred
    # logp = F.log_softmax(logits, dim=1)   # softmax over class dimension
    #
    # # Soft cross-entropy loss
    # loss = -(target.unsqueeze(1) * logp).sum(dim=1).mean()
    #
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("EFLOCO: Gaussian mode with sparse graphs is not supported in this implementation")
    _, points, adj_matrix, _ = batch

    # adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    time_epsilon = 1e-3
    # Original
    # t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)

    t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1.0 - time_epsilon)

    # Original
    # xt, epsilon = self.diffusion.sample(adj_matrix, t)

    # Modified
    z0 = torch.randn_like(adj_matrix.float())
    t_expand = t.view(-1, 1, 1).repeat(1, adj_matrix.shape[1], adj_matrix.shape[2])
    # Noise initialization
    xt = t_expand * adj_matrix + (1. - t_expand) * z0
    target = adj_matrix - z0

    # Original
    # t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    # epsilon_pred = self.forward(
    #     points.float().to(adj_matrix.device),
    #     xt.float().to(adj_matrix.device),
    #     t.float().to(adj_matrix.device),
    #     None,
    # )
    # epsilon_pred = epsilon_pred.squeeze(1)

    # Modified
    v_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        (t * 999).float().to(adj_matrix.device),
        None,
    )
    # Inspect gradient norms during training
    for name, param in self.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Param {name}: grad norm {grad_norm:.4e}")

    # Compute loss
    # loss = F.mse_loss(epsilon_pred, epsilon.float())
    loss = F.mse_loss(v_pred, target.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      # ts = self.diffusion.T + 1 - t
      x0_pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
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
        points = points.repeat(self.args.parallel_sampling, 1, 1)
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

      # steps = self.args.inference_diffusion_steps
      # time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
      #                                   T=self.diffusion.T, inference_T=steps)
      #
      # # Diffusion iterations
      # for i in range(steps):
      #   t1, t2 = time_schedule(i)
      #   t1 = np.array([t1]).astype(int)
      #   t2 = np.array([t2]).astype(int)
      #
      #   if self.diffusion_type == 'gaussian':
      #     xt = self.gaussian_denoise_step(
      #         points, xt, t1, device, edge_index, target_t=t2)
      #   else:
      #     xt = self.categorical_denoise_step(
      #         points, xt, t1, device, edge_index, target_t=t2)

      # New version

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
                  # pred = model(x, t * 999)  ### Copy from models/utils.py
                  # vec_t = torch.ones(shape[0], device=model.device) * t
                  batch_size, node_size, _ = adj_matrix.shape


                  drift = self.forward(
                      points,
                      xt,
                      vec_t * 999,
                      # None,
                      None,
                  )
                  drift = drift.squeeze(1)
                  # drift = to_flattened_numpy(drift)
                  # print(drift.shape)
                  xt = xt + drift * dt



      else:

          # from discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver
          from discrete_solver_Kinetic import MixtureDiscreteEulerSolver
          # from discrete_solver_Kinetic import KineticOptimalDiscreteEulerSolver
          #
          scheduler = PolynomialConvexScheduler(n=2.0)
          path = MixtureDiscreteProbPath(scheduler=scheduler)
          # wrapped_probability_denoiser = WrappedModel(model)

          # solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
          #                                     vocabulary_size=2)

          solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                              vocabulary_size=2)
          # x_init = torch.randint(size=(n_samples, dim), high=vocab_size, device=device)
          nfe = 64
          step_size = 1 / nfe

          n_plots = 9
          # epsilon = 1e-3
          linspace_to_plot = torch.linspace(0, 1 - 1e-3, n_plots)
          # linspace_to_plot = torch.linspace(0, 1, n_plots)
          sol = solver.sample(x_init=xt,
                              step_size=step_size,
                              verbose=False,
                              return_intermediates=True,
                              time_grid=linspace_to_plot,
                              eval_batch=batch)
          # sol = solver.sample(
          #     x_init=xt,
          #     step_size=step_size,
          #     verbose=False,
          #     return_intermediates=True,
          #     eval_batch=batch,
          #     use_step_optim=True,  # Enable optimized time steps
          #     nfe=50,  # Number of steps (e.g., 10 or 20 also work)
          #     step_optim_init_type='unif_t',  # Time-step initialization strategy
          #     time_grid=torch.tensor([0.0, 1 - 1e-3]),  # Sampling interval
          #
          # )
          xt = sol

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      tours, merge_iterations = merge_tours(
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )

      # Refine using 2-opt
      solved_tours, ns = batched_two_opt_torch(
          np_points.astype("float64"), np.array(tours).astype('int64'),
          max_iterations=self.args.two_opt_iterations, device=device)
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)
    # tours = np.concatenate(tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    # all_solved_costs = [tsp_solver.evaluate(tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)

    gap = (best_solved_cost - gt_cost) / gt_cost * 100

    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": ns,
        f"{split}/merge_iterations": merge_iterations,
        f"{split}/gap(%)": gap,
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

  # def get_edge_weights(self, adj_matrix):
  #   """
  #   Return per-class weights for nn.CrossEntropyLoss: [neg_weight, pos_weight]
  #   """
  #   num_pos = (adj_matrix == 1).sum().float() + 1e-8
  #   num_neg = (adj_matrix == 0).sum().float()
  #   total = num_pos + num_neg
  #
  #   pos_weight = num_neg / total  # Weight for edge-present class (1)
  #   neg_weight = num_pos / total  # Weight for edge-absent class (0)
  #
  #   # Return shape [2] for classes 0 and 1
  #   weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32, device=adj_matrix.device)
  #   return weights
  #
  # def get_edge_weights_decay(self, adj_matrix, epoch, decay_rate_pos=0.15, max_epoch=50):
  #   """
  #   Adjust positive/negative class weights by epoch: positive weight decays, negative stays fixed.
  #   decay_rate_pos: decay rate for positive-class weight (heuristic; typically < 1, e.g. 0.1)
  #   max_epoch: maximum training epochs (e.g. 50)
  #   """
  #
  #
  #   # Initial positive/negative class weights
  #   num_pos = (adj_matrix == 1).sum().float() + 1e-8
  #   num_neg = (adj_matrix == 0).sum().float()
  #   total = num_pos + num_neg
  #
  #   pos_weight = num_neg / total  # Weight for edge-present class (1)
  #   neg_weight = num_pos / total  # Weight for edge-absent class (0)
  #
  #   # Decay factor (increases with epoch)
  #   decay_factor = min(1.0, epoch / max_epoch)  # Keep in a safe range
  #
  #   # Positive weight decays but will not go below the negative weight
  #   new_pos_weight = pos_weight - decay_factor * decay_rate_pos * pos_weight
  #   if new_pos_weight < neg_weight:  # Stop when it reaches the negative weight
  #     new_pos_weight = neg_weight
  #
  #   # Return adjusted weights
  #   weights = torch.tensor([neg_weight, new_pos_weight], dtype=torch.float32, device=adj_matrix.device)
  #   return weights
  #
  # def get_edge_weights_decay_add(self, adj_matrix, epoch, decay_rate_pos=0.15, max_epoch=50):
  #     """
  #     Adjust positive/negative class weights by epoch: positive weight grows, negative stays fixed.
  #     decay_rate_pos: growth rate for positive-class weight (heuristic; typically < 1, e.g. 0.1)
  #     max_epoch: maximum training epochs
  #     """
  #
  #     # Initial positive/negative class weights
  #     num_pos = (adj_matrix == 1).sum().float() + 1e-8
  #     num_neg = (adj_matrix == 0).sum().float()
  #     total = num_pos + num_neg
  #
  #     pos_weight = num_neg / total  # Weight for edge-present class (1)
  #     neg_weight = num_pos / total  # Weight for edge-absent class (0)
  #
  #     # Growth factor (increases with epoch)
  #     growth_factor = min(1.0, epoch / max_epoch)  # Keep in a safe range
  #
  #     # Positive weight grows, capped to a maximum
  #     new_pos_weight = neg_weight + growth_factor * decay_rate_pos * neg_weight
  #     if new_pos_weight > 2.5:  # Avoid overly large positive weights
  #         new_pos_weight = 2.5
  #
  #     # Return adjusted weights
  #     weights = torch.tensor([neg_weight, new_pos_weight], dtype=torch.float32, device=adj_matrix.device)
  #     return weights
  #
  # def get_flowmatching_style_weights(self, adj_matrix, t):
  #     """
  #     t: float, time in flow matching, in [0, 1]
  #     """
  #     adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float32, device=self.device)
  #
  #     # Compute initial positive/negative counts
  #     num_pos = (adj_matrix == 1).sum().float() + 1e-8
  #     num_neg = (adj_matrix == 0).sum().float() + 1e-8
  #
  #     neg_weight = torch.tensor(1.0, device=adj_matrix.device)
  #
  #     # Initial positive weight
  #     init_pos_weight = (num_neg / num_pos).clamp(min=1.0)
  #
  #     # Positive weight increases with t
  #     pos_weight = init_pos_weight * t
  #     pos_weight = pos_weight.clamp(min=1.0, max=init_pos_weight)
  #     neg_weight = torch.ones_like(pos_weight)  # Match shape with pos_weight
  #
  #     weights = torch.stack([neg_weight, pos_weight])
  #
  #     return weights
  #
  # def compute_sample_weights(self, adj_matrix, t):
  #     """
  #     Adjust sample weights as a function of t (positive/negative weights vary)
  #     """
  #     # Negative weight is fixed
  #     neg_weight = 1.0  # Weight for negative samples
  #     # Initial positive weight
  #     initial_pos_weight = 1.0  # You may tune this
  #
  #     # Positive weight increases with t
  #     pos_weight = initial_pos_weight * (1 + t * 2)  # As t increases, pos weight increases
  #
  #     # Ensure positive weight is larger than negative weight
  #     pos_weight = torch.max(torch.tensor(pos_weight), torch.tensor(neg_weight + 0.1))  # pos > neg
  #
  #     # Create a tensor with the same shape as adj_matrix
  #     sample_weights = torch.ones_like(adj_matrix).float()  # Initialize to ones
  #     sample_weights[adj_matrix == 1] = pos_weight  # Assign weights to positive samples
  #     sample_weights[adj_matrix == 0] = neg_weight  # Assign weights to negative samples
  #
  #     return sample_weights





