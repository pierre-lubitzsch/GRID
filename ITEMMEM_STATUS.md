# Item memory vs plain TIGER on clean data: status

Last update: 2026-10-28 20:30. Detailed history: `WORKFLOW_PKM.md` (sections from 2026-10-27 on).

## Question

On clean data, does the item memory (im, imc) beat plain TIGER on test NDCG@10 across seeds,
datasets, tokenizers and training recipes, or was the single-seed +12% luck?

## Arms

| arm | what |
|---|---|
| plain | TIGER |
| im | + input-side item memory (p_drop 0.1, neighbour substitution p_nbr 0.2) |
| imc | im + consistency loss 0.1 |
| imfree | capacity control: same per-item vectors, no dropout, no neighbour substitution |

## Settings (label = run-name suffix `_ITEMMEM_<arch>clean<label>[_d<seed>]`)

| label | training recipe | tokenizer | jobs | status |
|---|---|---|---|---|
| `_es` | repo config = NAU's (Adam 1e-3, wd 1e-4, batch 1024, early stop on val R@5) | our RQ-VAE (`embeddings/<ds>_rqvae`) | `logs/itemmem_es_submit_20261027.tsv` | Beauty nearly done, Sports running, Toys queued |
| `_gprk` | GRID paper (Adam 5e-4, wd 1e-6, batch 256, early stop on val NDCG@10); plain/im/imc only | GRID RK-Means (NAU's SIDs) | `logs/itemmem_rk_submit_20261028.tsv` | CANCELLED 2026-10-29, replaced by `_gp40rk` (patience 40), logs/itemmem_gp40_submit_20261029.tsv |
| `_tigr` | TIGER paper (batch 256, lr 0.01 const 10k then inverse sqrt, 200k steps, dropout 0.1, 2000 user tokens) | our RQ-VAE | `logs/itemmem_tigr_submit_20261028.tsv` | pilot queued (Beauty s2 plain/im/imc), 24 chains HELD (plain/im/imc only) |
| `_tigraf` | TIGER paper with Adafactor (T5X default) | our RQ-VAE | `logs/itemmem_tigraf_submit_20261029.tsv` | pilot Beauty s2 plain/im/imc queued |
| `_tigrtok` | TIGER paper | TIGER's tokenizer | `logs/tigertok_jobs.txt` | PARKED (RQ-VAE collapses / diverges with Adagrad 0.4) |

All: datasets beauty / sports / toys, training seeds 2, 3, 5 (7, 11 deferred).
Launcher: `run_itemmem_grid.sh` (`RECIPE=grid|gridpaper|tiger`, `TOKENIZER=grid|rkmeans|tiger`,
`STAGE=train|cleaneval`, `CLEAN=1`). Results table: `python -m scripts.itemmem_clean_summary --recipe <label>`.
Release the held TIGER chains: `scontrol release $(cut -f4 logs/itemmem_tigr_submit_20261028.tsv | paste -sd,)`.

## Results so far

See `item_memory_cur.md` section 3 (Beauty and Sports `_es`, with corrected memory-off evals).

Bugs fixed 2026-10-28/29: memory-off evals crashed (unquoted braces in --wrap), then ran with the
memory ON (`eval_pathway:null` = YAML null); canonical value now `nulltoken`. Toys data was missing
in this tree (copied from the main tree, Toys `_es` resubmitted).

## Status 2026-11-21

All clean (5 seeds, both recipes) and poisoned ablation runs (6 architectures x 15 cells,
on/off at inference, before/F+R/NAU) are done. Nothing queued. See item_memory_cur.md section 6.

## Status 2026-11-24: Toys poisoned grid (second dataset)

Statistics for Beauty: `python -m scripts.itemmem_stats` (output in logs/itemmem_stats_20261123.txt);
slides 15/16 of slides/itemmem_clean_results.tex.

Toys poisoned grid: plain, im, imc, ims, imsc x 3 strata x 5 data seeds (plainscore skipped,
control already on Beauty). Data copied from the main tree (toys_spam_{,tgtmid_,tgtpopular_}seed<N>).
Launcher: `DS=toys` in run_item_memory_beauty.sh (labels/tags `ITEMMEM_toys_`, eval on toys);
submit: `bash scripts/submit_toys_grid.sh`; training pinned to H200 (`TRAIN_GRES`).
Pilot (mid seed 2, plain + imsc): train 11692081 / 11692083, chains 11692082 / 11692084.
Job list: logs/toys_grid_submit_<date>.txt. Tables: `python -m scripts.itemmem_ablation_table --ds toys`,
`python -m scripts.itemmem_stats --ds toys`.

### Queued 2026-11-24 (afternoon): MUNKEY grid, second attack, retrained references

Submit script `scripts/submit_munkey_ca_refs.sh`, job list `logs/munkey_ca_refs_submit_20261124.txt`.
All new training on H200 (`TRAIN_GRES`; the Beauty grid and the original MUNKEY cell were A100-80GB).
- MUNKEY (`run_munkey_beauty.sh`, now with DSEED/STRATUM, per-cell user keys): mk11 (best val N@10
  of mk00/11/22: .0343) on the 14 remaining Beauty cells; arms pre, zs, zsfull, full x F+R/NAU.
- Second attack: clone-append (Beauty, mid target, data seeds 2 3 5 7 11; data copied from the
  main tree), `ATTACK=clone_append`, models `<arch>_ca[_d<N>]`, archs plain im imc ims imsc.
- Retrained references (`STAGE=ref`, tags `ref_spam[_toys][_ca]_<stratum>_d<N>_<m>clean`): Beauty
  plain/imsc for mid d2/3/5, new `imclean` model for im on all 15 cells, clone-append x5;
  Toys `cleanref_{plain,im,imsc}` models x 15 cells.
Next once these are in: forget-only F+R (no retain data) on a subset of cells.
