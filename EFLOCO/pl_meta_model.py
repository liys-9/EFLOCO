

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info

from models.gnn_encoder import GNNEncoder
# from models.gnn_encoder_atsp_2 import GNNEncoder
# from models.gnn_encoder_atsp import GNNEncoder

from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion, GaussianDiffusion


class COMetaModel(pl.LightningModule):
  def __init__(self,
               param_args,
               node_feature_only=False):
    super(COMetaModel, self).__init__()
    self.args = param_args
    self.diffusion_type = self.args.diffusion_type
    self.diffusion_schedule = self.args.diffusion_schedule
    self.diffusion_steps = self.args.diffusion_steps
    self.sparse = self.args.sparse_factor > 0 or node_feature_only

    if self.diffusion_type == 'gaussian':
      out_channels = 1
      self.diffusion = GaussianDiffusion(
          T=self.diffusion_steps, schedule=self.diffusion_schedule)
    elif self.diffusion_type == 'categorical':
      out_channels = 2
      self.diffusion = CategoricalDiffusion(
          T=self.diffusion_steps, schedule=self.diffusion_schedule)
    else:
      raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

    self.model = GNNEncoder(
        n_layers=self.args.n_layers,
        hidden_dim=self.args.hidden_dim,
        out_channels=out_channels,
        aggregation=self.args.aggregation,
        sparse=self.sparse,
        use_activation_checkpoint=self.args.use_activation_checkpoint,
        node_feature_only=node_feature_only,
    )
    self.num_training_steps_cached = None

  # def on_test_start(self):
  #   self.test_metrics_cache = {
  #     "gt_cost": [],
  #     "gap": [],
  #     "merge_iterations": [],
  #     "2opt_iterations": [],
  #     "solved_cost": [],
  #   }

  def test_epoch_end(self, outputs):
    unmerged_metrics = {}
    for metrics in outputs:
      for k, v in metrics.items():
        if k not in unmerged_metrics:
          unmerged_metrics[k] = []
        unmerged_metrics[k].append(v)

    merged_metrics = {}
    for k, v in unmerged_metrics.items():
      merged_metrics[k] = float(np.mean(v))
    self.logger.log_metrics(merged_metrics, step=self.global_step)

  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.args.lr_scheduler == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
      # optimizer = torch.optim.AdamW(
      #     self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95),weight_decay=self.args.weight_decay)
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps(), warmup_steps=4000)(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }


  def duplicate_edge_index(self, edge_index, num_nodes, device):
    """Duplicate the edge index (in sparse graphs) for parallel sampling."""
    edge_index = edge_index.reshape((2, 1, -1))
    edge_index_indent = torch.arange(0, self.args.parallel_sampling).view(1, -1, 1).to(device)
    edge_index_indent = edge_index_indent * num_nodes
    edge_index = edge_index + edge_index_indent
    edge_index = edge_index.reshape((2, -1))
    return edge_index

  def train_dataloader(self):
    batch_size = self.args.batch_size
    train_dataloader = GraphDataLoader(
        self.train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=self.args.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True)
    return train_dataloader

  def test_dataloader(self):
    batch_size = 1
    # batch_size = self.args.batch_size
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

  def val_dataloader(self):
    batch_size = 8
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
    print("Validation dataset size:", len(val_dataset))
    val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_dataloader
