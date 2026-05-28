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

### Step 0b — Create a random 10% subset for fast end-to-end testing

Produces `src/data/erase_data/test_rsc15_seed_2/` by default. Samples 10% of
sessions; complete sessions are always kept intact (no partial sequences).

**Option A — from existing GRID dataset (recommended):**
Reads the already-converted TFRecords. Item IDs are kept identical, so the
existing SID tensor (`embeddings/rsc15/merged_predictions_tensor.pt`) can be
reused — no need to re-run Steps 1–2.

```bash
python -m src.data.erase_data.subsample_rsc15 \
    --from-grid-dir src/data/erase_data/rsc15
```

**Option B — from raw inter file:**
Runs the full conversion. Item IDs are remapped fresh over the subset, so a
new SID tensor is required (Steps 1–2 must be repeated for the subset).

```bash
python -m src.data.erase_data.subsample_rsc15 \
    --inter /home/pilu12/workspace/GRID/src/data/rsc15.inter
```

Both options use `--seed 2` and `--fraction 0.10` by default. To vary:

```bash
python -m src.data.erase_data.subsample_rsc15 \
    --from-grid-dir src/data/erase_data/rsc15 \
    --seed 42
# → src/data/erase_data/test_rsc15_seed_42/
```

When using Option A, downstream steps can reuse the full-dataset SID tensor:

```bash
SID=embeddings/rsc15/merged_predictions_tensor.pt  # same as full dataset

# Step 3: clean training on subset
sbatch run_tiger_train.sh test_rsc15_seed_2 clean "${SID}"

# Step 4: poisoning on subset
sbatch run_rsc15_poison.sh test_rsc15_seed_2
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
# Args: dataset  [poisoning_ratio]  (n_target_items via N_TARGET_ITEMS env var)
sbatch run_rsc15_poison.sh rsc15
sbatch run_rsc15_poison.sh rsc15 0.05   # pct5
```

Default parameters match ERASE: `seed=2`, `poisoning_ratio=0.01`, `n_target_items=10`, `placement=sprinkled`, `p_two_targets=0.119`.

The output directory name is derived automatically from the params:

```bash
N_TARGET_ITEMS=5 sbatch run_rsc15_poison.sh rsc15 0.02
# → src/data/erase_data/rsc15_spam_seed2_pct2_n5/
```

Outputs:
- `src/data/erase_data/rsc15_spam_seed<S>_pct<P>_n<N>/` — poisoned data
- `.../forget_manifest.json` — target items + spam user IDs
- `.../training_forget/` and `.../training_retain/` — forget/retain split

```bash
# Default params:
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10
```

---

## Step 5 — Train TIGER on poisoned data

```bash
# Args: dataset  clean|poison  semantic_id_path  [poisoning_ratio]  [n_target_items]
sbatch run_tiger_train.sh rsc15 poison "${SID}"
sbatch run_tiger_train.sh rsc15 poison "${SID}" 0.05       # pct5, n=10
sbatch run_tiger_train.sh rsc15 poison "${SID}" 0.05 5     # pct5, n=5
```

The script resolves the poisoned dataset directory from the ratio and n_target params (same logic as `run_rsc15_poison.sh`).

```bash
POISON_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt
```

This is the **starting point** for all unlearning algorithms.

---

## Step 6 — Unlearning

All unlearning algorithms accept the same base inputs: `<ckpt_path>`, `<data_dir>`, `[semantic_id_path]`. Extra Hydra overrides follow at the end.

```bash
# Shorthand used throughout this section (default params: seed=2, ratio=0.01, n=10)
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10
SID=embeddings/rsc15/merged_predictions_tensor.pt
```

For non-default poison params, pass `POISONING_RATIO` / `N_TARGET_ITEMS` as env vars — the unlearn script resolves the poisoned dataset path automatically:

```bash
POISONING_RATIO=0.05 N_TARGET_ITEMS=10 sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" ...
```

### 6a — SCIF (default, single-pass)

SCIF updates all TIGER parameters except the structural `bos_token` and `sep_token` by default (`unlearning.target_params=tiger`). To change which parameters are updated:

```bash
# default: all encoder/decoder/embedding weights, excluding bos_token and sep_token
unlearning.target_params=tiger

# all trainable parameters including bos_token and sep_token
unlearning.target_params=all

# only item SID embedding table + per-hierarchy decoder heads (fastest, narrowest)
unlearning.target_params=sid_embeddings

# only encoder sub-module weights
unlearning.target_params=encoder_only
```

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

### 6b — Sequential unlearning (per-request batches)

Processes forget requests in batches; runs post-eval automatically.

```
run_tiger_unlearn_sequential.sh <ckpt> <dataset> [algorithm] [sid] [neighborhood_aware] [batch_size] [sample_rate] [overrides...]
```

`[algorithm]` is optional at position 3, auto-detected by name (default: `scif`).
Can also be set via `UNLEARN_ALGORITHM=<algo>` env var.

#### Algorithm choices

| Algorithm | What it does | Key hyperparams |
|---|---|---|
| `scif` (default) | Single-shot Newton update via Conjugate Gradient | `damping`, `cg_max_iter`, `update_max_norm`, `target_params` |
| `unified` | Combined loss: L_retain + λ·L_forget + λ·L_sep | `lambda_forget`, `lambda_sep`, `unified_steps` / `n_batch_passes`, `unified_lr` |
| `finetune` | Continue training on retain-only data (Adam) | `finetune_steps`, `finetune_lr` |
| `neg_train` | Gradient ascent on forget + retain CE every N steps | `neg_train_steps`, `neg_train_lr`, `neg_retain_every` |
| `filter` | Decode-time output masking, no weight update | `filter_mode` (`global` \| `user_dependent`) |

#### One call per algorithm

```bash
# SCIF (default) — neighborhood-aware retain sampling recommended
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 scif "${SID}" true 8 1.0

# Unified objective
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 unified "${SID}" false 8 1.0 \
    unlearning.lambda_forget=1.0 unlearning.lambda_sep=0.1 unlearning.unified_steps=500

# Fine-tune on retain data
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 finetune "${SID}" false 8 1.0 \
    unlearning.finetune_steps=500 unlearning.finetune_lr=1e-3

# Negative training
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 neg_train "${SID}" false 8 1.0 \
    unlearning.neg_train_steps=200 unlearning.neg_train_lr=1e-3 unlearning.neg_retain_every=5

# Decode-time filter (global)
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 filter "${SID}" false 8 1.0 \
    unlearning.filter_mode=global
```

Mixed neighborhood for SCIF (half neighbourhood-aware, half uniform):

```bash
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" true 8 0.5
```

### 6c — Baseline algorithms

Single-pass via `run_tiger_unlearn_baselines.sh <algorithm> <ckpt> <data_dir> [sid] [overrides]`.
Sequential (per-request batches) via `run_tiger_unlearn_sequential.sh` with an `[algorithm]` arg — see §6b.

**Fine-tune on retain data:**

```bash
# Single-pass
sbatch run_tiger_unlearn_baselines.sh finetune \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.finetune_steps=500 unlearning.finetune_lr=1e-3

# Sequential (per-request batches)
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 finetune "${SID}" false 8 1.0 \
    unlearning.finetune_steps=500 unlearning.finetune_lr=1e-3
```

**Negative training:**

```bash
# Single-pass
sbatch run_tiger_unlearn_baselines.sh neg_train \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.neg_train_steps=200 unlearning.neg_train_lr=1e-3 \
    unlearning.neg_retain_every=5

# Sequential (per-request batches)
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 neg_train "${SID}" false 8 1.0 \
    unlearning.neg_train_steps=200 unlearning.neg_train_lr=1e-3
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
# Single-pass
sbatch run_tiger_unlearn_baselines.sh unified \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.forget_loss_level=token \
    unlearning.lambda_forget=1.0 \
    unlearning.lambda_sep=0.1 \
    unlearning.unified_steps=500 \
    unlearning.unified_lr=1e-4

# Sequential (per-request batches) — uses tiger_unlearn_unified_sequential config
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 unified "${SID}" false 8 1.0 \
    unlearning.lambda_forget=1.0 unlearning.lambda_sep=0.1 unlearning.unified_steps=500
```

Sequence-level forget loss:

```bash
sbatch run_tiger_unlearn_baselines.sh unified \
    "${POISON_CKPT}" "${POISON_DIR}" "${SID}" \
    unlearning.forget_loss_level=sequence
```

**Balanced forget/retain exposure (automatic).** Each optimizer step
gradient-accumulates `q_forget = ceil(n_forget/n_retain)` forget mini-batches
and `q_retain = ceil(n_retain/n_forget)` retain mini-batches (one is always 1).
Each sub-loss is averaged via `1/q_*` scaling, then a single `opt.step()` runs.
This guarantees that every forget sample and every retain sample contributes
to the gradient roughly the same number of times — independent of the relative
sizes of the two sets. The chosen `q_forget`/`q_retain` are logged at the
start of each call and recorded in the result dict.

**Sizing the loop.** Two equivalent knobs:

- `unlearning.unified_steps=N` — total optimizer updates (default 500).
- `unlearning.n_batch_passes=N` — total full passes over the batches.
  One pass = `min(n_forget_batches, n_retain_batches)` optimizer steps with
  balanced accumulation, so this expands to
  `unified_steps = N * min(n_forget, n_retain)`. Takes priority over
  `unified_steps` when set.

Per-step wall-time grows roughly with `q_forget + q_retain`. With a small
forget set (typical), `q_retain` ≈ `n_retain_batches / n_forget_batches`, so a
4-vs-51 imbalance makes each step ~13× heavier — prefer `n_batch_passes` to
stay step-count-aware of the imbalance, or lower `unified_steps` manually.

```bash
# Run 3 full passes through the batches (auto-scaled per request)
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 unified "${SID}" false 8 1.0 \
    unlearning.n_batch_passes=3 unlearning.lambda_forget=1.0 unlearning.lambda_sep=0.1
```

### 6e — Deletion specification

Controls what the algorithms treat as "known" and which items are used as neighborhood centers.

| `deletion_spec` | Known to algorithm | Neighborhood centers |
|---|---|---|
| `session` (default) | Spam user sessions | All distinct items in forget shards |
| `item` | Target items `I_f` from manifest | `I_f` only; non-target spam interactions allowed as repair rows |
| `item_pairs` | Target items `I_f` from manifest | `I_f` only (see §6g) |

```bash
# Item mode — attach to any algorithm via Hydra override
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" true \
    unlearning.deletion_spec=item
```

### 6g — Item unlearning (`deletion_spec=item_pairs`)

Surgically unlearns only the `(prefix → target_item)` training pairs from spam sessions rather than whole sessions.  For each spam session `[i_1,...,i_n]` and every position `j` where `i_j ∈ I_f`, one forget entry `[i_1,...,i_j]` is written.  The training collate expands this to the full-context pair `([i_1,...,i_{j-1}], i_j)` — the signal most responsible for the model predicting `i_j`.

Output: `training_forget_item_pairs/` sibling directory (created on first run, cached for re-use).

Requires `target_items` in `forget_manifest.json` (present in all bandwagon-poisoned datasets).

**Single-shot SCIF, spam sessions only:**

```bash
sbatch run_tiger_unlearn.sh "${POISON_CKPT}" "${POISON_DIR}" "${SID}" false \
    unlearning.deletion_spec=item_pairs
```

**Sequential SCIF, spam sessions only:**

```bash
# Use request_user_order=sorted because item_pairs entries have synthetic
# user_ids (0, 1, 2, …) that do not appear in the spam manifest.
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" false 8 1.0 \
    unlearning.deletion_spec=item_pairs \
    unlearning.request_user_order=sorted
```

With neighborhood-aware retain sampling:

```bash
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" true 8 1.0 \
    unlearning.deletion_spec=item_pairs \
    unlearning.request_user_order=sorted
```

**`--unlearn_whole_items`: global target suppression (spam + clean sessions)**

Adds `(prefix → i_f)` pairs from the clean retain set so target items are suppressed regardless of their context, not just in spam-induced predictions.

```bash
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 "${SID}" false 8 1.0 \
    unlearning.deletion_spec=item_pairs \
    unlearning.unlearn_whole_items=true \
    unlearning.request_user_order=sorted
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
| `GRID_POISON_DATA_DIR` | `src/data/erase_data/rsc15_spam_seed<S>_pct<P>_n<N>` | Poisoned dataset — derived from `POISON_SEED`, `POISONING_RATIO`, `N_TARGET_ITEMS` |
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

# SCIF target parameters (tiger = all except bos_token/sep_token, default for TIGER)
unlearning.target_params=tiger        # default
unlearning.target_params=all          # include bos_token and sep_token too
unlearning.target_params=sid_embeddings  # only embedding table + decoder heads
unlearning.target_params=encoder_only    # only encoder weights

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

# Test rsc15:
LLM_EMB_TEST=logs/inference/runs/2026-05-26/15-34-35/pickle/merged_predictions_tensor.pt

# Step 2: SID codebook
sbatch run_rkmeans_train.sh rsc15 "${LLM_EMB}"
RKMEANS_CKPT=logs/train/runs/<date>/<time>/checkpoints/last.ckpt
RKMEANS_CKPT_TEST=logs/train/runs/2026-05-26/16-10-46/checkpoints/checkpoint_000_000030.ckpt
sbatch run_rkmeans_inference.sh "${RKMEANS_CKPT}" rsc15 "${LLM_EMB}"
SID=embeddings/rsc15/merged_predictions_tensor.pt
SID_TEST=embeddings/test_rsc15_seed_2/merged_predictions_tensor.pt

# Step 3: clean training
sbatch run_tiger_train.sh rsc15 clean "${SID}"
CLEAN_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt

# Step 4: poison  (add arg 2 for non-default ratio, e.g. 0.05)
sbatch run_rsc15_poison.sh rsc15
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10  # adjust for non-default params

# Step 5: poisoned training  (add args 4/5 for non-default ratio/n_target)
sbatch run_tiger_train.sh rsc15 poison "${SID}"
POISON_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt

# Step 6: unlearn — set variables, then run
ALGO=scif                # scif | unified | finetune | neg_train | filter
NEIGHBORHOOD_AWARE=true  # true = neighborhood-aware retain sampling (recommended for scif)
BATCH_SIZE=8
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 \
    "${ALGO}" "${SID}" "${NEIGHBORHOOD_AWARE}" "${BATCH_SIZE}" 1.0

# Step 6 (alternative): item unlearning — only (prefix → target_item) pairs from spam sessions
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 \
    "${ALGO}" "${SID}" false "${BATCH_SIZE}" 1.0 \
    unlearning.deletion_spec=item_pairs \
    unlearning.request_user_order=sorted

# Step 6 (alternative): global item suppression — also unlearn target pairs from clean sessions
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 \
    "${ALGO}" "${SID}" false "${BATCH_SIZE}" 1.0 \
    unlearning.deletion_spec=item_pairs \
    unlearning.unlearn_whole_items=true \
    unlearning.request_user_order=sorted

# Step 7: evaluate
UNLEARN_CKPT=logs/unlearn/runs/<run_id>/checkpoints/unlearned.ckpt
sbatch run_tiger_eval_three_way.sh "${UNLEARN_CKPT}" "${CLEAN_CKPT}" "${POISON_CKPT}" "${SID}" src/data/erase_data/rsc15
```

## Typical rsc15 run sequence executed job ids

```bash
# One-time setup
python -m src.data.erase_data.convert_rsc15_inter --inter src/data/rsc15.inter --out-dir src/data/erase_data/rsc15

# Step 1: LLM embeddings (no SLURM job tracked; run completed 2026-05-21)
LLM_EMB=logs/inference/runs/2026-05-21/14-23-52/pickle/merged_predictions_tensor.pt

# Step 2: SID codebook
# 2a: train --- job 8978715
sbatch run_rkmeans_train.sh rsc15 "${LLM_EMB}"
RKMEANS_CKPT=logs/train/runs/2026-05-21/21-13-42/checkpoints/checkpoint_000_000030.ckpt
# 2b: inference — job 8985567
sbatch run_rkmeans_inference.sh "${RKMEANS_CKPT}" rsc15 "${LLM_EMB}"
SID=embeddings/rsc15/merged_predictions_tensor.pt

# Step 3: clean training --- job 8989546 (cancelled at 2-day time limit; last ckpt at step 2500)
# H200 training: job 9013420
sbatch run_tiger_train.sh rsc15 clean "${SID}"
CLEAN_CKPT=logs/train/runs/2026-05-22/16-08-24/checkpoints/checkpoint_epoch=000_step=002500.ckpt

# Step 4: poison --- job 8989494
sbatch run_rsc15_poison.sh rsc15
POISON_DIR=src/data/erase_data/rsc15_spam_seed2_pct1_n10

# Step 5: poisoned training --- job 9013396
# H200 training: job 9013421
sbatch run_tiger_train.sh rsc15 poison "${SID}"
POISON_CKPT=logs/train/runs/<date>/<time>/checkpoints/checkpoint_epoch=003.ckpt

# Step 6: unlearn --- not yet run
ALGO=scif                # scif | unified | finetune | neg_train | filter
NEIGHBORHOOD_AWARE=true
BATCH_SIZE=8
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT}" rsc15 \
    "${ALGO}" "${SID}" "${NEIGHBORHOOD_AWARE}" "${BATCH_SIZE}" 1.0

# Step 7: evaluate --- not yet run
UNLEARN_CKPT=logs/unlearn/runs/<run_id>/checkpoints/unlearned.ckpt
sbatch run_tiger_eval_three_way.sh "${UNLEARN_CKPT}" "${CLEAN_CKPT}" "${POISON_CKPT}" "${SID}" src/data/erase_data/rsc15
```

## Typical test_rsc15_seed_2 run sequence executed job ids

```bash
# Step 0b: subsample 10% from existing rsc15 GRID dataset
python -m src.data.erase_data.subsample_rsc15 --from-grid-dir src/data/erase_data/rsc15
# → src/data/erase_data/test_rsc15_seed_2/

# Step 1: LLM embeddings --- job 9015463
sbatch generate_embeddings.sh test_rsc15_seed_2
LLM_EMB_TEST=logs/inference/runs/2026-05-26/15-34-35/pickle/merged_predictions_tensor.pt

# Step 2: SID codebook
# 2a: train --- job 9017156 (failed: DDP find_unused_parameters, fixed in rkmeans_train_flat.yaml), job 9017715 (success)
sbatch run_rkmeans_train.sh test_rsc15_seed_2 "${LLM_EMB_TEST}"
RKMEANS_CKPT_TEST=logs/train/runs/2026-05-26/16-10-46/checkpoints/checkpoint_000_000030.ckpt
# 2b: inference --- job 9018131
sbatch run_rkmeans_inference.sh "${RKMEANS_CKPT_TEST}" test_rsc15_seed_2 "${LLM_EMB_TEST}"
SID_TEST=embeddings/test_rsc15_seed_2/merged_predictions_tensor.pt

# Step 3: clean training --- job 9018466
sbatch run_tiger_train.sh test_rsc15_seed_2 clean "${SID_TEST}"
CLEAN_CKPT_TEST=logs/train/runs/2026-05-26/16-22-13/checkpoints/<latest>.ckpt

# Step 4: poison --- job 9019345
sbatch run_rsc15_poison.sh test_rsc15_seed_2          # default pct1_n10
# sbatch run_rsc15_poison.sh test_rsc15_seed_2 0.05   # example: pct5
POISON_DIR_TEST=src/data/erase_data/test_rsc15_seed_2_spam_seed2_pct1_n10  # adjust for non-default params

# Step 5: poisoned training ---
# pct 0.01, ntarget 10: job 9019894
# pct 0.05, ntarget 10: job 9056493
# pct 0.1, ntarget 10: job 9056494
# Add args 4/5 for non-default ratio/n_target: sbatch run_tiger_train.sh test_rsc15_seed_2 poison "${SID_TEST}" 0.05 10
sbatch run_tiger_train.sh test_rsc15_seed_2 poison "${SID_TEST}"
POISON_CKPT_TEST=logs/train/runs/2026-05-26/16-42-29/checkpoints/<latest>.ckpt

# Step 6: unlearn --- not yet run
ALGO=scif                # scif | unified | finetune | neg_train | filter
NEIGHBORHOOD_AWARE=true  # true = neighborhood-aware retain sampling (recommended for scif)
BATCH_SIZE=8
sbatch run_tiger_unlearn_sequential.sh "${POISON_CKPT_TEST}" test_rsc15_seed_2 \
    "${ALGO}" "${SID_TEST}" "${NEIGHBORHOOD_AWARE}" "${BATCH_SIZE}" 1.0

# SCIF, no NAU, BS 256, pct 1: 9055237
# SCIF, NAU, BS 256, pct 1: 9058181

# SCIF subselection, no NAU, BS 256, pct 1: 9081530
# SCIF subselection, NAU, BS 256, pct 1: 9081531

# SCIF subselection, no NAU, n_unlearning_chunks 10, pct 1: 9081752
# SCIF subselection, NAU, n_unlearning_chunks 10, pct 1: 9081753

# neg_train, no NAU, BS 256, pct 1: 9061644
# neg_train, NAU, BS 256, pct 1: 9061645

# unified, no NAU, BS 256, pct 1: 9068699
# unified, NAU, BS 256, pct 1: 9068700

# unified, no NAU, n_unlearning_chunks 10, pct 1: 9081740
# unified, NAU, n_unlearning_chunks 10, pct 1: 9081741

# unified, no NAU, n_unlearning_chunks 10, pct 1, n_batch_passes 4, lambda_forget 0.1: 9088422
# unified, NAU, n_unlearning_chunks 10, pct 1, n_batch_passes 4, lambda_forget 0.1: 9088423

# unified, no NAU, n_unlearning_chunks 10, pct 1, n_batch_passes 4, lambda_forget 1.0: 9088500
# unified, NAU, n_unlearning_chunks 10, pct 1, n_batch_passes 4, lambda_forget 1.0: 9088501


# Step 7: evaluate --- not yet run
UNLEARN_CKPT_TEST=logs/unlearn/runs/<run_id>/checkpoints/unlearned.ckpt
sbatch run_tiger_eval_three_way.sh "${UNLEARN_CKPT_TEST}" "${CLEAN_CKPT_TEST}" "${POISON_CKPT_TEST}" "${SID_TEST}" src/data/erase_data/test_rsc15_seed_2
```
