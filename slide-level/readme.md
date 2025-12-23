## Multi-task Aggregator for slide-level diagnosis

A lightweight README for training and evaluating the multi-task aggregator on whole-slide level tasks.

## Dataset / CSV format

- Organize per-slide labels in a CSV file (example location: `csvs/test.csv`).
- The training CSV must include a `task_id` column that indicates which task the row belongs to. `task_id` indexes into the `tasks` list in the config.

Minimal example (CSV):

```csv
name,code,[task_name],task_id
path1/slide1,slide1,1(label),1
path1/slde2,slide2,1,0
path2/slide3,slide3,0,1
```

Explanation:

- name: unique path to slide/feature file depending on your loader.
- code: code for each slide (you can also replace the code as name if your code is the slide path).
- `task_id`: integer index of the task. In our settings, `task_id=0` -> `cancer`, `1` -> `candidiasis`, `2` -> `cluecell`.
- [task_name]: target label for this slide and this task (use the label encoding your dataset expects).

## Configuration

The default config is `configs/multitask_unicas.yaml`. Important keys:

- `base_path`: base path for CSV files (e.g. `./csvs`).
- `train_csv`, `valid_csv`, `test_csv`: filenames under `base_path`.
- `tasks`: list of task names. Order must match how you use `task_id` in the CSV.
- `num_tasks`: number of tasks.
- `img_batch`: number of patches sampled per slide.
- encoder: encoder to train the aggregator.
- `embed_dim`, `depth`, `aggregator`: model architecture hyperparameters.
- `loss_fns`, `loss-weights`: loss choices and weighting.

Edit the YAML to match your dataset and training preferences.

## Encode

```
python encode.py --encoder UniCAS --csv csvs/test.csv
```

## Train

Use PyTorch's `torchrun` to launch training. Example single-node single-process run:

```powershell
torchrun --nproc_per_node=1 --master_port=2252 train_distribute.py --config configs/multitask_unicas.yaml
```

For multi-GPU training, set `--nproc_per_node` to the number of processes (GPUs) you want to use, and adjust `--master_port` if needed.

## Evaluate

To run evaluation / inference only, add the `--eval-only` flag:

```powershell
torchrun --nproc_per_node=1 --master_port=2252 train_distribute.py --config configs/multitask_unicas.yaml --eval-only
```

## Notes & tips

- `task_id` should be consistent between CSV and `configs/*.yaml`.
- If you have pretrained aggregator weights, set `weights` in the YAML.
- To freeze the  task, set its learning rate in `lr_head` to a very small value (<1e-8).
- `cont` in the config controls whether to log false predictions for certain cancer grades (project-ASC-US+ behavior).
