# EFLOCO

EFLOCO is a discrete flow matching approach for **few-step solution generation** on combinatorial optimization problems.

This codebase is a cleaned/rebranded research implementation. The directory layout and several core components follow the open-source DIFUSCO project (Sun & Yang, NeurIPS 2023).



Create the conda environment:

```bash
conda env create -f efloco/environment.yml
conda activate efloco
```

Build the Cython extension used to merge TSP heatmaps into tours:

```bash
cd efloco/utils/cython_merge
python setup.py build_ext --inplace
cd -
```

Run training (from repo root):

```bash
python efloco/train.py --task tsp --storage_path ./outputs --do_train
```



## Tasks

### TSP

TSP uses `efloco/pl_tsp_model_consin.py` (`TSPModel`) by default in `efloco/train.py`.

Useful flags (see `efloco/train.py`):

- `--training_split`: path to the training instances (default: `data/tsp/tsp50_train_concorde.txt`)
- `--validation_split`: path to validation instances (default: `data/tsp/tsp50_test_concorde.txt`)
- `--test_split`: path to test instances
- `--sparse_factor`: enable sparse graphs when > 0 (use with care; not all branches are production-hardened)
- `--save_numpy_heatmap`: save the predicted heatmap as `.npy` during evaluation

Example:

```bash
python efloco/train.py \
  --task tsp \
  --storage_path ./outputs \
  --diffusion_type categorical \
  --batch_size 64 \
  --num_epochs 50 \
  --do_train
```

## Data

- `data/`: helper scripts and datasets used in experiments.
- `data/tsp/*.txt`: text-format splits used by `--training_split/--validation_split/--test_split`.

For exact reproduction commands, see `reproducing_scripts.md`.



To evaluate with a checkpoint:

```bash
python efloco/train.py --task tsp --storage_path ./outputs --ckpt_path /path/to.ckpt --do_test
```

## Codebase map

- `efloco/pl_tsp_model_consin.py`: EFLOCO TSP Lightning module used by default
- `efloco/pl_tsp_model.py`: older/experimental TSP module (kept for reference)
- `efloco/discrete_solver_Kinetic_optimed.py`: discrete samplers/solvers
- `efloco/utils/`: schedulers, step optimization, and utilities
- `efloco/utils/cython_merge/`: Cython extension for merging tours from heatmaps

## Acknowledgements

- **Discrete Flow Matching (DFM) implementation references**:
  - [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching/tree/main)
  - [RobinKa/discrete-flow-matching-pytorch](https://github.com/RobinKa/discrete-flow-matching-pytorch/tree/master)
  - [lebellig/discrete-fm](https://github.com/lebellig/discrete-fm/tree/master)
- **Optimized time steps (ATSS)**: our implementation is inspired by the time-step optimization idea in Xue et al. (CVPR 2024) ([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Accelerating_Diffusion_Sampling_with_Optimized_Time_Steps_CVPR_2024_paper.pdf)).

## Notes on optimized time steps (Xue et al., CVPR 2024)

- **Following the idea in Xue'24**: we use the reparameterization \(\lambda=\log(\alpha/\sigma)\) and solve a time-step optimization problem with `scipy.optimize.minimize(..., method="trust-constr")` (framework-level).


## Citation


