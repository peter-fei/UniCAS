# Multi-task Aggregator for Slide-Level Diagnosis

This repository contains the implementation of a Multi-task Aggregator for whole-slide image (WSI) diagnosis. It supports multi-task learning scenarios (e.g., cancer screening, candidiasis testing, and clue cell diagnosis).

## Prerequisites & Directory Structure

To ensure the code runs correctly, your data directory must follow a specific structure due to the loader implementation.

1. **Feature Directory**: You must have a directory in the project root matching the pattern `Pathology_{encoder}_p*` (e.g., `Pathology_unicas_patch`).
2. **Slide Data**: Inside this directory, each slide's features must be stored in a `torch/images.pt` file.

**Recommended Structure:**

```text
UniCAS/
├── Pathology_unicas_patch/       # Feature root directory
│   ├── slide1/                   # Slide name (must match 'name' in CSV)
│   │   └── torch/
│   │       └── images.pt         # Feature tensor
│   ├── slide2/
│   │   └── torch/
│   │       └── images.pt
│   └── ...
├── csvs/                         # CSV files location
│   ├── train.csv
│   └── test.csv
├── configs/
│   └── multitask_unicas.yaml
└── ...
```

## Dataset Preparation

### CSV Format

Organize your slide labels in CSV files (e.g., `csvs/train.csv`, `csvs/test.csv`).

**Requirements:**

- **Training CSV**: Must include a `task_id` column indicating the active task for that row.
- **Columns**: `name`, `code`, `task_id`, and columns for each specific task label.

**Minimal Example:**

```csv
name, code, cancer, candidiasis, cluecell, task_id
slide1, slide1, 1, 0, 1, 0
slide2, slide2, 1, 0, 0, 1
slide3, slide3, 0, 1, 0, 2
```

**Column Explanation:**

- `name`: Unique identifier for the slide. This **must match** the folder name in your feature directory.
- `code`: Alternative identifier (can be the same as name).
- `[task_name]` (e.g., `cancer`, `candidiasis`): Ground truth labels. You must have a column for every task defined in your config.
- `task_id`: Integer index for the active task in the current row.
  - Example mapping: `0` -> `cancer`, `1` -> `candidiasis`, `2` -> `cluecell`.

## Configuration

The default configuration file is `configs/multitask_unicas.yaml`.

**Key Parameters:**

- `base_path`: Directory containing CSV files (default: `./csvs`).
- `train_csv` / `valid_csv` / `test_csv`: Filenames relative to `base_path`.
- `tasks`: List of task names (e.g., `['cancer', 'candidiasis', 'cluecell']`). Order matters as it corresponds to `task_id`.
- `num_tasks`: Total number of tasks.
- `img_batch`: Number of patches to sample per slide during training.
- `encoder`: Name of the encoder (used to locate the `Pathology_{encoder}_p*` directory).
- `embed_dim`: Embedding dimension (Note: Ensure this matches your encoder output).

## Usage. Encode Features (Optional)

If you need to extract features from slides:

```bash
python encode.py --encoder UniCAS --csv csvs/test.csv
```

### 2. Training

Use `torchrun` for distributed training.

```powershell
torchrun --nproc_per_node=1 --master_port=2252 train_distribute.py --config configs/multitask_unicas.yaml
```

### 3. Evaluation

To run inference/evaluation only (skips training):

```powershell
torchrun --nproc_per_node=1 --master_port=2252 train_distribute.py --config configs/multitask_unicas.yaml --eval-only
```
