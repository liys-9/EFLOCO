# Reproduce Results

## Training

### Training on TSP50

```bash


python -u efloco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp50" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp50_train_concorde.txt" \
  --validation_split "/your/tsp50_valid_concorde.txt" \
  --test_split "/your/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 2
```



### Training on TSP500

```bash

python -u efloco/train.py \
  --task "tsp" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_concorde.txt" \
  --validation_split "/your/tsp500_valid_concorde.txt" \
  --test_split "/your/tsp500_test_concorde.txt" \
  --sparse_factor 50 \
  --batch_size 8 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 2
```



### Evaluation on TSP50-Categorical with greedy decoding

```bash


python -u efloco/train.py \
  --task "tsp" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/efloco" \
  --training_split "data/tsp/tsp50_train_concorde.txt" \
  --validation_split "/data/tsp/tsp50_test_concorde.txt" \
  --test_split "/data/tsp/tsp50_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 2 \
  --ckpt_path "" \
  --resume_weight_only
```