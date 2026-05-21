# RSC15 Training & Unlearning Workflow

End-to-end guide: clean-data training → bandwagon poisoning → TIGER training on poisoned data → unlearning with all supported algorithms → evaluation.

All commands are run from the repo root. Every `sbatch` script also works as a plain `bash` locally (remove `sbatch`); SLURM headers are ignored by bash.

---

## Prerequisites

| Artifact | Default path |
|---|---|
| `rsc15.inter` (ERASE RecBole format) | `src/data/rsc15.inter` |
| Flan-T5-XL weights (HuggingFace) | Downloaded on first run |
| SLURM partitions | `gpu` (single A100/H200), `pgpu` (2× H200) |

---

## Step 0 — Convert rsc15 to GRID TFRecord layout

Only needed once per dataset. Produces `training/`, `evaluation/`, `testing/`, `items/` under `src/data/erase_data/rsc15/`.

```bash
python -m src.data.erase_data.convert_rsc15_inter \
    --inter src/data/rsc15.inter \
    --out-dir src/data/erase_data/rsc15
```

Optional flags: `--seed 2 --min-session-length 2 --rows-per-shard 5000`

For a fast smoke-test subset (first 5 000 sessions):

```bash
python -m src.data.erase_data.convert_rsc15_inter \
    --inter src/data/rsc15.inter \
    --out-dir src/data/erase_data/rsc15_smoke \
    --max-sessions 5000
```

---

## Step 1 — Generate flan-T5 item embeddings

Required by **both** the RKMeans codebook training (Step 2) and embedding-based neighborhood construction (Step 6).

```bash
sbatch generate_embeddings.sh rsc15
```

- Runs flan-T5-XL inference over `src/data/erase_data/rsc15/items/`.
- Merges pickle shards and saves `{"embeddings": (N, 2048), "item_ids": (N,)}` to `logs/inference/runs/<date>/<time>/pickle/merged_predictions_tensor.pt`.
- `item_ids` stores raw rsc15 IDs (up to ~214M); the indexed format lets downstream code handle non-sequential IDs correctly.
- Prints the install command at the end:

```
Install for training: bash scripts/install_semantic_id_tensor.sh rsc15 <pickle>/merged_predictions_tensor.pt
```

Save the pickle path — you'll need it for Step 2 and for `--embedding_path` in unlearning.

```bash
# Keep a stable reference copy
LLM_EMB=logs/inference/runs/<date>/<time>/pickle/merged_predictions_tensor.pt
```

---

## Step 2 — RKMeans codebook (semantic IDs)

### 2a — Train codebooks

```bash
sbatch run_rkmeans_train.sh rsc15 "${LLM_EMB}"
```

Trains 3 codebooks of width 256 on the flan-T5 embeddings.

### 2b — Run inference to assign SIDs

```bash
RKMEANS_CKPT=logs/train/runs/<date>/<time>/checkpoints/last.ckpt

sbatch run_rkmeans_inference.sh "${RKMEANS_CKPT}" rsc15 "${LLM_EMB}"
```

Outputs `merged_predictions_tensor.pt` — a `(4, N)` int tensor (3 codebook digits + 1 de-duplication digit) and installs it automatically:

```
embeddings/rsc15/merged_predictions_tensor.pt   ← GRID_SEMANTIC_ID_PATH
```

This is the **SID tensor** used by all training and unlearning steps. Keep it stable.

```bash
SID=embeddings/rsc15/merged_predictions_tensor.pt
```

---

## Step 3 — Train TIGER on clean data

```bash
sbatch run_tiger_train.sh rsc15 clean "${SID}"
```

Checkpoint lands at `logs/train/runs/<date>/<time>/checkpoints/`.

```bash
CLEAN_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt
```

This is the **reference** model for computing relative utility in evaluation.

---

## Step 4 — Bandwagon poisoning

Creates the poisoned dataset and the forget/retain split in one script.

```bash
sbatch run_rsc15_poison.sh rsc15
```

Default parameters match ERASE: `seed=2`, `poisoning_ratio=0.01`, `n_target_items=10`, `placement=sprinkled`, `p_two_targets=0.119`.

Override via environment:

```bash
POISON_SEED=42 POISONING_RATIO=0.02 N_TARGET_ITEMS=5 sbatch run_rsc15_poison.sh rsc15
```

Outputs:
- `src/data/erase_data/rsc15_spam_seed2_pct1_n10/` — poisoned data
- `.../forget_manifest.json` — target items + spam user IDs
- `.../training_forget/` and `.../training_retain/` — forget/retain split

```bash
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10
```

---

## Step 5 — Train TIGER on poisoned data

```bash
sbatch run_tiger_train.sh rsc15 poison "${SID}"
```

```bash
POISON_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt
```

This is the **starting point** for all unlearning algorithms.

---

## Step 6 — Unlearning

All unlearning algorithms accept the same base inputs: `<ckpt_path>`, `<data_dir>`, `[semantic_id_path]`. Extra Hydra overrides follow at the end.

```bash
# Shorthand used throughout this section
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10
SID=embeddings/rsc15/merged_predictions_tensor.pt
```

### 6a — SCIF (default, single-pass)

```bash
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" false
```

With neighborhood-aware retain sampling (prefix, recommended):

```bash
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" true \
    unlearning.neighbor_aware_factor=8 \
    unlearning.neighborhood_method=prefix \
    unlearning.sid_prefix_length=2
```

With embedding-based neighborhood (requires LLM embeddings from Step 1):

```bash
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" true \
    unlearning.neighborhood_method=embedding \
    "unlearning.embedding_path='${LLM_EMB}'" \
    unlearning.embedding_epsilon=5.0 \
    unlearning.embedding_max_neighbors=100
```

### 6b — SCIF sequential (per-request batches)

Processes forget requests in batches; runs post-eval automatically.

```bash
# Args: ckpt dataset semantic_id neighborhood_aware batch_size sample_rate [overrides]
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" true 8 1.0
```

Mixed neighborhood (half neighborhood, half uniform):

```bash
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" true 8 0.5
```

### 6c — Baseline algorithms

All via `run_tiger_unlearn_baselines.sh <algorithm> <ckpt> <data_dir> [sid] [overrides]`.

**Fine-tune on retain data:**

```bash
sbatch run_tiger_unlearn_baselines.sh finetune \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.finetune_steps=500 unlearning.finetune_lr=1e-3
```

**Negative training:**

```bash
sbatch run_tiger_unlearn_baselines.sh neg_train \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.neg_train_steps=200 unlearning.neg_train_lr=1e-3 \
    unlearning.neg_retain_every=5
```

**Decode-time filter (no weight update):**

```bash
# global: mask all target items for every user
sbatch run_tiger_unlearn_baselines.sh filter \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.filter_mode=global

# user_dependent: mask only items from a user's removed interactions
sbatch run_tiger_unlearn_baselines.sh filter \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.filter_mode=user_dependent
```

**Retrain upper bound** (train from scratch on retain-only data):

```bash
# Retrain on poisoned-minus-forget = effectively clean; no unlearn entry point
sbatch run_tiger_train.sh rsc15 clean "${SID}"
```

### 6d — Unified objective (L_retain + λ L_forget + λ L_sep)

```bash
sbatch run_tiger_unlearn_baselines.sh unified \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.forget_loss_level=token \
    unlearning.lambda_forget=1.0 \
    unlearning.lambda_sep=0.1 \
    unlearning.unified_steps=500 \
    unlearning.unified_lr=1e-4
```

Sequence-level forget loss:

```bash
sbatch run_tiger_unlearn_baselines.sh unified \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.forget_loss_level=sequence
```

### 6e — Deletion specification

Controls what the algorithms treat as "known" and which items are used as neighborhood centers.

| `deletion_spec` | Known to algorithm | Neighborhood centers |
|---|---|---|
| `session` (default) | Spam user sessions | All distinct items in forget shards |
| `item` | Target items `I_f` from manifest | `I_f` only; non-target spam interactions allowed as repair rows |

```bash
# Item mode — attach to any algorithm via Hydra override
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" true \
    unlearning.deletion_spec=item
```

### 6f — Step 4 local repair (optional, off by default)

Attach to unified or neg_train runs:

```bash
sbatch run_tiger_unlearn_baselines.sh unified \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.local_repair.enabled=true \
    unlearning.local_repair.logit_suppression=true \
    unlearning.local_repair.gamma=1.0
```

Available sub-flags: `logit_suppression`, `adapter_repair`, `prefix_repair`, `mass_regularization`.

---

## Step 7 — Evaluation

All unlearn runs produce `checkpoints/unlearned.ckpt` under `logs/unlearn/runs/<run_id>/`.

```bash
UNLEARN_CKPT=logs/unlearn/runs/<run_id>/checkpoints/unlearned.ckpt
```

### 7a — Single checkpoint eval (clean test set)

```bash
sbatch run_tiger_eval.sh "${UNLEARN_CKPT}" src/data/erase_data/rsc15 "${SID}"
```

Outputs `logs/eval/runs/<date>/<time>/csv/version_0/metrics.csv`.

### 7b — Three-way eval (unlearned vs clean ref vs poisoned)

```bash
sbatch run_tiger_eval_three_way.sh \
    "${UNLEARN_CKPT}" \
    "${CLEAN_CKPT}" \
    "${POISON_CKPT}" \
    "${SID}" \
    src/data/erase_data/rsc15
```

### 7c — Batch sweep eval (all unlearn runs at once)

```bash
sbatch run_tiger_eval_batch_sweep.sh
```

Evaluates every `unlearned.ckpt` found under `logs/unlearn/runs/`. Used as input to the table collector.

### 7d — Forget-target recall probe (↓ = better unlearning)

```bash
python -m scripts.eval_forget_targets \
    --data_dir "${POISON_DIR}" \
    --ckpt_path "${UNLEARN_CKPT}"
```

Reports `forget_recall@10`: how often the model still retrieves the removed target items.

### 7e — Collect comparison table (all algorithms)

After running the batch sweep (7c), compare all algorithms with the clean reference:

```bash
bash run_collect_unlearn_eval_table.sh \
    logs/eval/batch_sweep/<stamp> \
    logs/unlearn/runs
```

Produces a CSV table under `logs/eval/collected/<stamp>/` with columns for relative utility, forgetting metrics, and per-run metadata.

---

## Key paths reference

| Variable | Default path | Description |
|---|---|---|
| `GRID_DATA_DIR` | `src/data/erase_data/rsc15` | Clean dataset |
| `GRID_POISON_DATA_DIR` | `src/data/erase_data/rsc15_spam_seed2_pct1_n10` | Poisoned dataset |
| `GRID_SEMANTIC_ID_PATH` / `SID` | `embeddings/rsc15/merged_predictions_tensor.pt` | RKMeans SID tensor (D×N int) |
| `LLM_EMB` | `logs/inference/runs/.../pickle/merged_predictions_tensor.pt` | flan-T5 LLM embeddings (N×2048 float, indexed dict) |
| Train ckpts | `logs/train/runs/<date>/<time>/checkpoints/` | |
| Unlearn ckpts | `logs/unlearn/runs/<run_id>/checkpoints/unlearned.ckpt` | |

`GRID_DATA_DIR`, `GRID_POISON_DATA_DIR`, and `GRID_SEMANTIC_ID_PATH` are set by `scripts/resolve_grid_dataset.sh` and can be overridden as environment variables before calling any script.

---

## Common Hydra overrides

```bash
# Change retain budget (default: 16 × |D_f| rows)
unlearning.retain_samples_used_for_update=32

# Cap total retain rows regardless of multiplier
unlearning.retain_max_rows=1000

# Target only embedding layers (faster, less aggressive)
unlearning.target_params=embedding

# Adjust SCIF CG convergence
unlearning.cg_max_iter=500 unlearning.damping=0.001

# Fix seed
seed=42 UNLEARN_SEED=42
```

---

## Typical rsc15 run sequence (copy-paste)

```bash
# One-time setup
python -m src.data.erase_data.convert_rsc15_inter --inter src/data/rsc15.inter --out-dir src/data/erase_data/rsc15

# Step 1: LLM embeddings (submit, note job ID)
sbatch generate_embeddings.sh rsc15
LLM_EMB=logs/inference/runs/<date>/<time>/pickle/merged_predictions_tensor.pt

# Step 2: SID codebook
sbatch run_rkmeans_train.sh rsc15 "${LLM_EMB}"
RKMEANS_CKPT=logs/train/runs/<date>/<time>/checkpoints/last.ckpt
sbatch run_rkmeans_inference.sh "${RKMEANS_CKPT}" rsc15 "${LLM_EMB}"
SID=embeddings/rsc15/merged_predictions_tensor.pt

# Step 3: clean training
sbatch run_tiger_train.sh rsc15 clean "${SID}"
CLEAN_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt

# Step 4: poison
sbatch run_rsc15_poison.sh rsc15
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10

# Step 5: poisoned training
sbatch run_tiger_train.sh rsc15 poison "${SID}"
POISON_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt

# Step 6: unlearn (SCIF + neighborhood)
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" true

# Step 7: evaluate
UNLEARN_CKPT=logs/unlearn/runs/<run_id>/checkpoints/unlearned.ckpt
sbatch run_tiger_eval_three_way.sh "${UNLEARN_CKPT}" "${CLEAN_CKPT}" "${POISON_CKPT}" "${SID}" src/data/erase_data/rsc15
```
