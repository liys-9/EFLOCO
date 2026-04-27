"""Lightning module for the EFLOCO TSP model."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info


from co_datasets.tsp_graph_dataset import TSPGraphDataset
# from co_datasets.tsp_graph_dataset_Copy1 import TSPGraphDataset
from pl_meta_model import COMetaModel
# from utils.diffusion_schedulers import InferenceSchedule
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

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)



  def threshold_based_f(self, x_t_plus_delta, t_plus_delta, t_segment_end, logits_p1_given_xt_plus_delta, x_at_t_segment_end, threshold):
    if (threshold, int) and threshold == 0:
      return x_at_t_segment_end
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    less_than_threshold = t_plus_delta < threshold
    from discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver
    solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                        vocabulary_size=2, sparse=self.sparse)

    # print('less_than_threshold',less_than_threshold.shape)
    x = solver.jump_state_to(x_t_plus_delta, t_plus_delta, t_segment_end, logits_p1_given_xt_plus_delta)
    # print('x',x.shape)
    less_than_threshold = less_than_threshold.view(-1, 1, 1) #no spr
    # less_than_threshold = less_than_threshold.view(-1, 1)
    res = (
        less_than_threshold * solver.jump_state_to(x_t_plus_delta, t_plus_delta, t_segment_end, logits_p1_given_xt_plus_delta)
        + (~less_than_threshold) * x_at_t_segment_end
    )
    return res

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, x_1, _ = batch
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
      x_1 = route_edge_flags.reshape((batch_size, num_edges // batch_size))

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(x_1.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    use_step_optim = self.current_epoch >= 5  # Enable StepOptim after this epoch

    if use_step_optim:
        if not hasattr(self, 't_grid') or self.t_grid is None:
            from utils.step_optim import StepOptim
            from discrete_solver_Kinetic_optimed import get_or_load_t_grid
            scheduler = PolynomialConvexScheduler(n=2.0)
            self.path = MixtureDiscreteProbPath(scheduler=scheduler)
            self.t_grid = get_or_load_t_grid(
                scheduler=scheduler,
                nfe=20,  # Number of steps to simulate
                eps=1 - 1e-3,
                initType='unif_t',
                device=x_1.device
            )

        # Uniformly sample t within intervals from the optimized grid
        def sample_time_uniform_interval(t_grid, batch_size, device):
            indices = torch.randint(0, len(t_grid) - 1, (batch_size,), device=device)
            t_left = t_grid[indices]
            t_right = t_grid[indices + 1]
            return t_left + torch.rand_like(t_left) * (t_right - t_left)

        t = sample_time_uniform_interval(self.t_grid, x_1.shape[0], x_1.device)
        seg_indices = torch.searchsorted(self.t_grid, t, side="left").clamp(
          min=1)  # .clamp(min=1) prevents the inclusion of 0 in indices.
        t_segment_end = self.t_grid[seg_indices]

    else:
        # Early training: sample t from a uniform distribution
        time_epsilon = 1e-3
        t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)
        segments = torch.linspace(0, 1, 20, device=x_1.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(
          min=1)  # .clamp(min=1) prevents the inclusion of 0 in indices.
        t_segment_end = segments[seg_indices]



    # segment_ends_expand = segment_ends.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])

    max_eps = t_segment_end - t - 1e-6
    # con_delta = 1e-3
    delta = torch.full_like(max_eps, 1e-3)
    min_eps = torch.zeros_like(max_eps)
    delta = torch.clamp(delta, min=min_eps, max=max_eps)
    t_plus_delta = torch.clamp(t + delta, max=1.0)

    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    time_epsilon = 1e-3
    x_1 = x_1.long()
    x_0 = torch.randn_like(x_1.float())
    x_0 = (x_0 > 0).long()


    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = path_sample.x_t.float()

    path_sample_r = path.sample(t=t_plus_delta, x_0=x_0, x_1=x_1)
    x_t_plus_delta = path_sample_r.x_t.float()

    path_sample_seg = path.sample(t=t_segment_end, x_0=x_0, x_1=x_1)
    x_at_t_segment_end = path_sample_seg.x_t.float()
    # print('x_at_segment_ends',x_at_segment_ends.shape)


    if self.sparse:

      t_ = t.reshape(-1, 1).repeat(1, x_1.shape[1]).reshape(-1)
      r_ = t_plus_delta.reshape(-1, 1).repeat(1, x_1.shape[1]).reshape(-1)
      segment_ends_ = t_segment_end.reshape(-1, 1).repeat(1, x_1.shape[1]).reshape(-1)
      x_t = x_t.reshape(-1)
      x_t_plus_delta = x_t_plus_delta.reshape(-1)
      x_1 = x_1.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(x_1.device).reshape(2, -1)

    rng_state = torch.cuda.get_rng_state(self.device)
    # print(adj_matrix.shape, xt.shape,points.shape, t.shape)
    # x0_pred = self.forward(
    #     points.float().to(adj_matrix.device),
    #     xt.float().to(adj_matrix.device),
    #     t_.float().to(adj_matrix.device),
    #     edge_index,
    # )
    # print('xt',xt.shape)
    logits_p1_given_xt = self.forward(
        points.float().to(x_1.device),
        x_t.float().to(x_1.device),
        t.float().to(x_1.device),
        edge_index,
    )
    # print('xtP',x0_pred.shape)

    torch.cuda.set_rng_state(rng_state,self.device) # Shared Dropout Mask
    with torch.no_grad():
      # x0_pred_r = self.forward(
      #   points.float().to(adj_matrix.device),
      #   xr.float().to(adj_matrix.device),
      #   r_.float().to(adj_matrix.device),
      #   edge_index,
      # )
      logits_p1_given_xt_plus_delta = self.forward(
        points.float().to(x_1.device),
        x_t_plus_delta.float().to(x_1.device),
        t_plus_delta.float().to(x_1.device),
        edge_index,
      )
      logits_p1_given_xt_plus_delta = torch.nan_to_num(logits_p1_given_xt_plus_delta)

    from discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver
    solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                        vocabulary_size=2, sparse= self.sparse)


    x_jump_from_t = solver.jump_state_to(x_t, t, t_segment_end, logits_p1_given_xt)
    threshold = 0.95
    # print('jump')
    x_jump_from_t_plus_delta = self.threshold_based_f(
      x_t_plus_delta, t_plus_delta, t_segment_end, logits_p1_given_xt_plus_delta, x_at_t_segment_end, threshold)
    ft = F.one_hot(x_jump_from_t.to(torch.int64), num_classes=2).float()
    fr = F.one_hot(x_jump_from_t_plus_delta.to(torch.int64), num_classes=2).float()
    loss_mse = F.mse_loss(ft, fr)
    loss_func = nn.CrossEntropyLoss().to(device=x_1.device)
    loss = loss_func(logits_p1_given_xt, x_1.long())
    total_loss = loss + (0.1 *loss_mse)

    #
    self.log("train/loss", total_loss, prog_bar=True)
    self.log("mse", loss_mse, prog_bar=True)
    self.log("cross", loss, prog_bar=True)
    return total_loss

  # def gaussian_training_step(self, batch, batch_idx):
  #   if self.sparse:
  #     raise ValueError("EFLOCO: Gaussian mode with sparse graphs is not supported in this implementation")
  #   _, points, adj_matrix, _ = batch

  #   # adj_matrix = adj_matrix * 2 - 1
  #   adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
  #   # Sample from diffusion
  #   time_epsilon = 1e-3


  #   t = torch.rand(adj_matrix.shape[0], device=adj_matrix.device) * (1.0 - time_epsilon)




  #   z0 = torch.randn_like(adj_matrix.float())
  #   t_expand = t.view(-1, 1, 1).repeat(1, adj_matrix.shape[1], adj_matrix.shape[2])

  #   xt = t_expand * adj_matrix + (1. - t_expand) * z0
  #   target = adj_matrix - z0


  #   v_pred = self.forward(
  #       points.float().to(adj_matrix.device),
  #       xt.float().to(adj_matrix.device),
  #       (t * 999).float().to(adj_matrix.device),
  #       None,
  #   )



  #   loss = F.mse_loss(v_pred, target.float())
  #   self.log("train/loss", loss)
  #   return loss

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

          from discrete_solver_Kinetic_optimed import MixtureDiscreteEulerSolver

          #
          scheduler = PolynomialConvexScheduler(n=2.0)
          path = MixtureDiscreteProbPath(scheduler=scheduler)


          solver = MixtureDiscreteEulerSolver(model=self.forward, path=path,
                                              vocabulary_size=2,sparse= self.sparse)

          

          n_plots = 9
          # epsilon = 1e-3
          linspace_to_plot = torch.linspace(0, 1 - 1e-3, n_plots)

          sol = solver.sample(
              x_init=xt,
              verbose=False,
              return_intermediates=True,
              eval_batch=batch,
              use_step_optim=True,  # Enable optimized time steps
              nfe=2,  # Number of function evaluations / steps
              step_optim_init_type='unif_t',  # Time-step initialization strategy
              time_grid=torch.tensor([0.0, 1 - 1e-3]),  # Sampling interval

          )
          xt = sol
      #xt = xt.reshape(-1)
      # print(xt,xt.shape)

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

      # print(ns)

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

  





