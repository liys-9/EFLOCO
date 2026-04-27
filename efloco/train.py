"""EFLOCO: training and evaluation entrypoint (PyTorch Lightning).

Run from repository root, e.g.:
  python efloco/train.py --task tsp --storage_path ...
The `efloco/` package directory is on sys.path so legacy flat imports work
(`from pl_tsp_model_consin import ...`).
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EFLOCO_PACKAGE_DIR = _REPO_ROOT / "efloco"
for _p in (str(_EFLOCO_PACKAGE_DIR), str(_REPO_ROOT)):
  if _p not in sys.path:
    sys.path.insert(0, _p)

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pl_tsp_model_consin import TSPModel


def arg_parser():
  parser = ArgumentParser(
      description='Train/evaluate EFLOCO (PyTorch Lightning) on combinatorial optimization datasets.'
  )
  parser.add_argument('--task', type=str, required=True)
  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50-50_concorde.txt')
  parser.add_argument('--validation_examples', type=int, default=64)

  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lr_scheduler', type=str, default='constant')

  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='gaussian')
  parser.add_argument('--diffusion_schedule', type=str, default='linear')
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_schedule', type=str, default='linear')
  parser.add_argument('--inference_trick', type=str, default="ddim")
  parser.add_argument('--sequential_sampling', type=int, default=16)
  parser.add_argument('--parallel_sampling', type=int, default=1)

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  parser.add_argument('--project_name', type=str, default='efloco')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default=None)
  parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_valid_only', action='store_true')

  args = parser.parse_args()
  return args


def main(args):
  epochs = args.num_epochs
  set_random_seed()
  if args.task != 'tsp':
    raise NotImplementedError("Only --task tsp is supported in this EFLOCO release.")

  model_class = TSPModel
  saving_mode = 'min'

  model = model_class(param_args=args)

  tb_logger = TensorBoardLogger(
    save_dir=os.path.join(args.storage_path, f'models-new-100nodes'),
    name=args.resume_id
  )

  checkpoint_callback = ModelCheckpoint(
      monitor='val/solved_cost', mode=saving_mode,
      save_top_k=10, save_last=True,
      dirpath=os.path.join(tb_logger.save_dir, 'checkpoints'),
  )

  lr_callback = LearningRateMonitor(logging_interval='step')

  trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
      logger=tb_logger,
      check_val_every_n_epoch=1,
      strategy=DDPStrategy(static_graph=False),
      precision=32 if args.fp16 else 32,
      accumulate_grad_batches=16
  )
  print(torch.cuda.current_device(), torch.cuda.get_device_name())

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path)

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_callback.best_model_path)

  elif args.do_test:
    trainer.validate(model, ckpt_path=ckpt_path)
    if not args.do_valid_only:
      trainer.test(model, ckpt_path=ckpt_path)
  trainer.logger.finalize("success")


import random
import numpy as np
import time


def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def set_random_seed(max_seed=9999):
  seed = int(time.time()) % max_seed
  print(f"[INFO] Using random seed: {seed}")

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True

  return seed


if __name__ == '__main__':
  _args = arg_parser()
  main(_args)
