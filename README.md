# LoRA Teacher

LoRA Teacher is a research codebase for parameter-efficient adaptation of decoder-only language models. The repository centers on a two-stage workflow:

1. Train a temporary LoRA teacher on a task.
2. Consolidate the teacher's useful behavior back into a base model with selective module updates, attribution, replay, and optional memory-aware extensions.

The release in this folder is a cleaned GitHub-ready version built from the original `/root/lora-teacher` workspace. It keeps the core code layout and the commands you were actually using, while removing generated artifacts, caches, machine-specific clutter, and one-off utilities.

## Repository Layout

```text
.
├── configs/
│   ├── data/
│   ├── experiments/
│   ├── method/
│   ├── model/
│   └── train/
├── dataset/
├── scripts/
├── src/
│   └── cl_lora_probe/
├── README.md
├── USAGE.md
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

Main code areas:

- `src/cl_lora_probe/modeling`: base model, LoRA teacher, selective student.
- `src/cl_lora_probe/runners`: train/eval pipeline entrypoints.
- `src/cl_lora_probe/consolidation`: attribution-guided consolidation logic.
- `src/cl_lora_probe/memory`: memory tag, replay, and subspace-bank components.
- `src/cl_lora_probe/ttt`: trigger-based test-time tuning.
- `configs`: runnable YAML configs.
- `scripts`: convenience entrypoints for the workflows used in practice.

## What Was Cleaned

Removed from the release version:

- `__pycache__`, `.pyc`, `.DS_Store`
- auto-generated sweep configs under `configs/experiments/sweeps/generated/`
- one-off helper scripts that were not part of the maintained workflow
- hardcoded evaluation scratch code with machine-specific paths

Kept intentionally:

- the full `src/cl_lora_probe` package
- the config hierarchy under `configs/`
- the commands and workflows reflected in your original usage notes

## Environment

The original runtime environment name is `lora_teacher`.

### Option 1: Use the provided Conda environment file

```bash
conda env create -f environment.yml
conda activate lora_teacher
pip install -e .
```

### Option 2: Install into an existing environment

```bash
conda activate lora_teacher
pip install -r requirements.txt
pip install -e .
```

Notes:

- `lm-eval` is included because `scripts/evaluate_tasks.py` depends on it.
- If your model tokenizer requires extra backend packages, install them in the same environment.

## Quick Start

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd lora-teacher
```

### 2. Activate the environment

```bash
conda activate lora_teacher
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

### 3. Set the base model path

Edit one of these files before running:

- `configs/model/qwen3_0_6b_local.yaml`
- `configs/model/base_model.yaml`
- `configs/experiments/continual_arc_easy_sst2_cola_all_in_one.yaml`
- `configs/experiments/independent_lora_merge_qwen3_8b_8task.yaml`

Replace placeholder values like `/absolute/path/to/...` with the real local path to your HuggingFace-compatible model directory.

### 4. Prepare datasets

Dataset paths in this release are repo-relative examples under `dataset/`. The code supports:

- local `arrow`
- local `parquet`
- local `json` / `jsonl`
- local `csv`
- Hugging Face datasets via `dataset_name`

See [dataset/README.md](dataset/README.md) for the expected layout and field format.

### 5. Run the single-task pipeline

For the GSM8K-style local Qwen setup:

```bash
bash scripts/run_single_task_gsm8k_qwen3_local.sh
```

You must first set at least these environment variables:

```bash
export MODEL_PATH=/absolute/path/to/Qwen3-0.6B
export TRAIN_PARQUET_PATH=/absolute/path/to/gsm8k_train.parquet
```

For the generic config-based pipeline:

```bash
bash scripts/run_single_task.sh
```

## Core Workflows

### Single-task training and consolidation

This reproduces the main teacher-then-consolidation pipeline:

```bash
python3 -m cl_lora_probe.runners.run_single_task_pipeline \
  --model_config configs/model/qwen3_0_6b_local.yaml \
  --lora_config configs/model/lora.yaml \
  --data_config configs/data/gsm8k_parquet_local.yaml \
  --teacher_train_config configs/train/teacher_gsm8k_quick.yaml \
  --consolidation_train_config configs/train/consolidation_gsm8k_quick.yaml \
  --attribution_config configs/method/attribution_gsm8k.yaml \
  --selector_config configs/method/selector_gsm8k.yaml \
  --loss_config configs/method/losses_gsm8k.yaml \
  --output_dir outputs/qwen3_gsm8k_stage1 \
  --baseline full_method
```

### Consolidation only

If you already have a trained teacher adapter:

```bash
python3 -m cl_lora_probe.runners.train_consolidation \
  --model_config configs/model/qwen3_0_6b_local.yaml \
  --lora_config configs/model/lora.yaml \
  --data_config configs/data/gsm8k_parquet_local.yaml \
  --attribution_config configs/method/attribution_gsm8k.yaml \
  --selector_config configs/method/selector_gsm8k.yaml \
  --loss_config configs/method/losses_gsm8k.yaml \
  --train_config configs/train/consolidation_gsm8k_quick.yaml \
  --teacher_adapter_dir outputs/qwen3_gsm8k_stage1/teacher_stage/teacher_lora \
  --output_dir outputs/qwen3_gsm8k_stage1/consolidation_stage_resume
```

### Trigger-based TTT

```bash
python3 -m cl_lora_probe.runners.run_ttt_eval \
  --model_config configs/model/sst2_consolidated_student.yaml \
  --data_config configs/data/sst2_arrow.yaml \
  --train_config configs/train/consolidation_gsm8k_quick.yaml \
  --ttt_config configs/experiments/ttt_eval_sst2_single_task.yaml \
  --memory_artifacts_config configs/experiments/memory_artifacts_sst2_single_task.yaml \
  --output_dir outputs/sst2_ttt_eval
```

### Continual experiment

All-in-one continual config:

```bash
bash scripts/run_continual_arc_easy_sst2_cola.sh
```

Or explicitly:

```bash
python3 scripts/run_continual_sweep.py \
  --base_config configs/experiments/continual_arc_easy_sst2_cola_all_in_one.yaml \
  --sweep_config configs/experiments/sweeps/continual_arc_easy_sst2_cola_hparam_sweep.yaml \
  --optimizer adamw
```

### Independent LoRA merge experiment

```bash
python3 -m cl_lora_probe.runners.run_independent_lora_merge_experiment \
  --experiment_config configs/experiments/independent_lora_merge_qwen3_8b_8task.yaml
```

### Task evaluation with lm-eval

```bash
python3 scripts/evaluate_tasks.py \
  --base_model /absolute/path/to/model_or_checkpoint \
  --tasks arc_easy,sst2,rte,mrpc,qnli \
  --batch_size 64 \
  --output_json results/eval/results.json
```

Optional LoRA adapter:

```bash
python3 scripts/evaluate_tasks.py \
  --base_model /absolute/path/to/base_model \
  --lora_path /absolute/path/to/teacher_lora \
  --tasks mrpc \
  --batch_size 16 \
  --output_json results/eval/mrpc.json
```

## Dataset Configuration

The data loader lives in `src/cl_lora_probe/data/task_dataset.py`.

Supported input styles:

- prompt/target pairs
- instruction/input/output triples
- raw `text`
- task-specific templates such as `gsm8k`, `sst2`, `rte`, `mrpc`, `qnli`, `arc`, `boolq`, `hellaswag`, `piqa`, `mbpp`

Important config fields:

- `data_files`: split-to-path mapping for local files
- `dataset_name`: Hugging Face dataset name if loading from the hub
- `template_style`: selects the formatter
- `max_length`
- `answer_only_loss`
- `calibration_samples`
- `use_train_as_anchor`

If a split is missing, the loader can fall back automatically:

- missing `validation`: uses a subset of train
- missing `test`: reuses validation
- missing `calibration`: uses a subset of train
- missing `anchor` and `use_train_as_anchor: true`: uses a subset of train

## Output Artifacts

Typical single-task outputs include:

- `teacher_stage/teacher_lora/`
- `consolidation_stage/consolidated_student/`
- `consolidation_stage/attribution_report.json`
- `consolidation_stage/selection_report.json`
- `pipeline_summary.json`
- `configs_snapshot/`

Memory-aware runs may also save:

- `memory_tags.json`
- `tag_buffer.json`
- `subspace_bank.pt`
- `replay_capture_report.json`
- `ttt_summary.json`

## Notes

- This repository assumes you already have local access to the base model weights.
- Paths in the example configs are meant to be edited before training.
- The script names were preserved where possible for compatibility with your original workflow, even if some file names still reflect older experiment naming.

## Usage Reference

For the cleaned version of your original command history, see [USAGE.md](USAGE.md).
