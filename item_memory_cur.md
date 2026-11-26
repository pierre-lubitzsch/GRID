# Item memory: current state (2026-11-05)

Companion files: `ITEMMEM_STATUS.md` (job bookkeeping), `WORKFLOW_PKM.md` (full history).

## 1. The four methods

All four share the plain TIGER backbone: T5 encoder-decoder, 4 + 4 layers, 6 heads of dimension
64, d_model 128, MLP 1024, ReLU. Each history item enters as its 4 semantic-ID tokens. The decoder
generates the next item's 4 tokens; beam search is used at inference. The methods differ only in
what is added to the encoder input, and in one extra loss for imc.

| method | what is added | extra loss | extra parameters |
|---|---|---|---|
| **plain** | nothing (TIGER) | none | none |
| **im** | item memory with pathway dropout and neighbor substitution | none | one 128-d vector per catalog item (~1.5M on Beauty) + adapter (128x128) + null token |
| **imc** | as im | consistency loss, weight 0.1 | as im |
| **imfree** | the same per-item vectors, but no dropout and no neighbor substitution (capacity control) | none | as im (null token frozen, unused) |

### Item memory (im, imc, imfree)

- **Table:** a learnable vector `v_i` per catalog item, initialized N(0, 0.02).
- **How it is read:** for every item in the user's history, `adapter(v)` (a 128x128 linear map) is
  added to all 4 of that item's SID-token embeddings, before the encoder. The decoder and the
  output are unchanged: no reranking in these four arms.
- **What v is during training (im, imc)**, drawn per item occurrence:
  - with probability 0.1 (`p_drop`): a learned **null token**. This is pathway dropout: the model
    must also predict without the item's entry.
  - with probability 0.2 (`p_nbr`): a **neighbor mix**. It is the softmax(sim / 0.1)-weighted
    average of the item's K' nearest neighbors' entries, with K' ~ U{1..4}. Neighbors come from
    cosine similarity of frozen flan-T5-XL content embeddings, and the mix is detached (no gradient
    to the neighbors). This makes an entry a swappable carrier of item-specific information
    rather than an arbitrary bias.
  - otherwise: the item's own entry `v_i`.
- **At evaluation:** always `v_i`.
- **imfree:** `p_drop = p_nbr = 0`, so always `v_i`. It has the same capacity as im without the
  training design. This tests whether im's gain is just extra parameters.

### The adapter

- **Definition:** `adapter(v) = W v + b`, one linear layer (128 x 128 + bias, 16.5k parameters,
  default PyTorch init). It is **shared by all items** (MUNKEY's `h_psi`).
- **Role:** it maps the memory vectors into the SID-token embedding space. The table stores the
  per-item content; `W` decides how that content enters the encoder.
- **Gradient:** it is applied to whatever `v` was drawn (own entry, null token or neighbor mix),
  so it gets a gradient on every occurrence. That includes neighbor mixes, which are themselves
  detached.
- **Start:** entries ~ N(0, 0.02) and a zero null token, so the memory's first contribution is
  small.
- **Memory off:** the bias `b` is added to every history item. Memory off
  (`eval_pathway=null`) feeds the null token through the adapter, which is exactly the "dropped"
  input of training for im / imc. For imfree the null token is frozen at 0, so memory off adds
  `b` alone. `eval_pathway=skip` adds nothing at all.

### Training step of im

1. Sample a batch of histories and draw `v` for every history item (own entry / null / neighbor mix).
2. Add `adapter(v)` to the item's 4 SID token embeddings; run the unchanged encoder and decoder.
3. Loss: only the plain TIGER loss (cross-entropy on the next item's 4 SID tokens).
4. One Adam step on everything jointly, from scratch: backbone, table, adapter, null token, with
   the same lr and schedule as plain.

The own entry `v_i` gets a gradient only when it was drawn (about 70% of occurrences). Only the
input side uses the memory: the target item is never looked up.

### Consistency loss (imc only)

A second forward pass is run with every entry replaced by its deterministic full-K (K = 4)
neighbor average (detached). The loss is KL(stopgrad(p_own) || p_neighbors), summed over the 4
SID levels, with weight 0.1. It trains the backbone so that "entry replaced by its neighbors"
predicts like the item's own entry. The point is that an entry can later be changed or deleted
without breaking predictions. It costs a second forward pass per step.

### Related arms not in this clean study

- **imsc** (the PKM paper's poisoned-data headline): im + an output-side item score that reranks
  the final beams + consistency.
- **ims**: im + the output score.
- **plainscore**: plain + the score head only.
- **category memory**: tested 10-27, small gain alone, hurts on top of im.

## 2. How the models are trained (settings)

Every setting trains on **clean** data (no spam users): Beauty, Sports and Toys,
training seeds 2, 3, 5. Test N@10 is measured on the checkpoint with the best validation score.

| label | recipe | lr / schedule | wd | batch | length / selection | tokenizer | other |
|---|---|---|---|---|---|---|---|
| `_es` | repo config = NAU's | Adam 1e-3 constant | 1e-4 | 1024 (2 GPUs x 32 x 16) | early stop: 10 validations x 100 steps, val Recall@5 | our RQ-VAE on flan-T5-XL (4 learned levels, latent 64) | dropout 0.15, no user token |
| `_gprk` | GRID paper (arXiv 2507.22224) | Adam 5e-4 | 1e-6 | 256 (2 x 32 x 4) | early stop: 10 x 100 steps, val NDCG@10 | GRID RK-Means | as `_es`; paper reports Beauty N@10 .0347 |
| `_tigr` | TIGER paper (arXiv 2305.05065) | Adam 0.01 for 10k steps, then inverse sqrt | 1e-4 | 256 (2 x 32 x 4) | 200k steps (Toys 100k), no early stop; best-val and final ckpt | our RQ-VAE | dropout 0.1, 2000 hashed user tokens; paper reports Beauty .0384 |
| `_tigraf` | TIGER paper with Adafactor (the T5X default optimizer for this lr/schedule) | Adafactor 0.01 const 10k, then inverse sqrt | 0 | 256 | as `_tigr` | our RQ-VAE | pilot Beauty s2 queued (2026-10-29) |
| `_tigrtok` | TIGER paper | | | | | TIGER's tokenizer | PARKED: Adagrad 0.4 collapses / diverges |

Notes:
- Our `_es` recipe is the repo's config, not the GRID paper's. `_gprk` is the GRID paper's.
- The TIGER paper does not name its optimizer; we use Adam.
- The item memory's neighbors always come from the flan-T5-XL embeddings, whichever tokenizer the
  SIDs come from.

Cut on 2026-10-28 (user): `_esrk` (NAU recipe on RK-Means; differs from `_es` only in the
tokenizer) and imfree in the new settings (`_gprk`, `_tigr`). imfree stays in `_es` for all datasets.

## 3. Finished runs (clean data, test NDCG@10, best-val checkpoint)

Setting `_es` (repo/NAU recipe, our RQ-VAE tokenizer). "Off" = the item memory replaced by its
null token at test time (memory switched off).

Beauty:

| method | s2 | s3 | s5 | mean | vs plain (paired) | memory off (s2/s3/s5) |
|---|---|---|---|---|---|---|
| plain | .0322 | .0328 | .0333 | .0327 | | |
| im | .0366 | .0376 | .0372 | .0371 | **+13.5%** (+13.9 / +14.7 / +11.8) | .0260 / .0295 / .0261 |
| imc | .0384 | .0390 | .0366 | .0380 | **+16.1%** (+19.4 / +18.8 / +9.9) | .0288 / .0292 / .0297 |
| imfree | .0330 | .0314 | .0315 | .0320 | -2.4% (+2.7 / -4.3 / -5.5) | .0095 / .0055 / .0107 |

Sports:

| method | s2 | s3 | s5 | mean | vs plain (paired) | memory off (s2/s3/s5) |
|---|---|---|---|---|---|---|
| plain | .0149 | .0176 | .0167 | .0164 | | |
| im | .0185 | .0166 | .0185 | .0179 | +9.4% (+23.8 / -6.1 / +10.6) | .0120 / .0106 / .0128 |
| imc | .0178 | .0174 | .0185 | .0179 | +9.6% (+19.3 / -1.4 / +10.8) | .0124 / .0119 / .0133 |
| imfree | .0135 | .0129 | .0130 | .0131 | -19.5% (-9.5 / -27.0 / -22.1) | .0049 / .0041 / .0057 |

Toys:

| method | s2 | s3 | s5 | mean | vs plain (paired) | memory off (s2/s3/s5) |
|---|---|---|---|---|---|---|
| plain | .0301 | .0307 | .0312 | .0306 | | |
| im | .0377 | .0348 | .0372 | .0366 | **+19.4%** (min +13.4) | .0224 / .0223 / .0236 |
| imc | .0368 | .0373 | .0388 | .0376 | **+22.8%** (min +21.4) | .0235 / .0242 / .0229 |
| imfree | .0328 | .0320 | .0322 | .0324 | +5.6% (min +3.4) | .0084 / .0055 / .0050 |

Reading:
- **The gain holds on all three datasets.** Beauty: im +13.5%, imc +16.1%; Toys: im +19.4%, imc +22.8%;
  every seed positive on both. Sports: about +9.5%, but seed 3 is slightly below plain for both.
- **imc vs im:** slightly ahead on Beauty (+16.1 vs +13.5%), equal on Sports.
- **imfree is far behind im/imc** (-2.4%, -19.5%, +5.6% on Toys): the gain is mostly the memory's
  training design, not capacity.
- **The models depend on the memory:** switched off, im / imc fall 23 to 35% and land below plain.
  imc loses less than im (consistency makes it more robust to a missing entry, as intended); imfree
  collapses without its entries (it never learned to work without them).
- Reference: TIGER paper Beauty .0384, GRID paper .0347.

Two bugs found in the inherited eval code (both fixed 2026-10-28/29): the memory-off evals first
crashed (unquoted braces in `sbatch --wrap`), then silently ran with the memory ON (`null` parsed as
YAML null). The numbers above are from the corrected evals.

### Earlier poisoned-data results (PKM paper grid, for context)

On the spam-poisoned paper grid (3 target strata x 5 data seeds), after unlearning, **imsc** beats
plain on N@10 by +19.5% (Forget+Repair) and +19.8% (NAU), at least +8% in every cell, with spam
removed equally. Those models used a 4,000-step training cap. That cap is why the clean study
re-trains with early stopping.

### First `_gprk` result (2026-10-29)

Beauty plain s5: test N@10 **.0244** (val .0342), best checkpoint at step 5,700, stopped at 6,700.
That is the paper's rule taken literally (patience 10 validations of 100 optimizer steps at batch 256),
but the val curve was still rising slowly. 1,000 steps at batch 256 are 4x fewer samples than our
`_es` patience at batch 1024. The GRID paper reports .0347 for RK-Means; our val (.034) matches it,
our test does not. The comparison inside `_gprk` (plain vs im/imc) is still paired and valid.
**User decision 2026-10-29: resubmit with patience 40** (label `_gp40rk`, `GP_PATIENCE=40`, 27 trainings + evals in `logs/itemmem_gp40_submit_20261029.tsv`). All `_gprk` jobs cancelled; plain s5 kept for reference only.
Note: the paper selects the checkpoint by val Recall@10 and early-stops on NDCG@10; we use NDCG@10 for both.

### Setting `_gp40rk` (GRID paper recipe, RK-Means, patience 40), 2026-11-05

| dataset | method | s2 | s3 | s5 | mean | vs plain (paired) | memory off |
|---|---|---|---|---|---|---|---|
| Beauty | plain | .0302 | .0363 | .0351 | .0339 | | |
| Beauty | im | .0406 | .0390 | .0407 | .0401 | **+19.2%** (min +7.5) | .0318 / .0301 / .0326 |
| Beauty | imc | .0396 | .0410 | .0410 | .0405 | **+20.3%** (min +13.0) | .0325 / .0340 / .0336 |
| Toys | plain | .0296 | .0304 | .0303 | .0301 | | |
| Toys | im | .0384 | .0381 | .0381 | .0382 | **+26.9%** (min +25.5) | .0274 / .0282 / .0262 |
| Toys | imc | .0394 | .0378 | .0368 | .0380 | **+26.2%** (min +21.3) | .0296 / .0291 / .0286 |
| Sports | plain | resuming | .0189 | .0187 | | | |
| Sports | im | .0210 | .0210 | resuming | | +10.9% (s3 only) | .0150 / .0144 / - |
| Sports | imc | resuming | .0207 | resuming | | +9.6% (s3 only) | - / .0154 / - |

Patience 40 fixed the early stop: Beauty plain .0339 (the paper reports .0347 for RK-Means).
Four Sports runs hit the 36 h limit (Sports trains for 25k to 50k steps); resumed 2026-11-05
(`EXTEND_FROM`, jobs 11447702/04/06/08). The eval then picks the best checkpoint across both segments.

### TIGER-recipe pilots (Beauty s2, 200k steps)

Bug found 2026-11-05: a resumed segment writes to a new run dir, and Lightning resets
ModelCheckpoint's best score when the dirpath changes, so train.py's test after a resume used
the best of the LAST segment only. New `scripts/best_ckpt_across_segments.py`; cleaneval now
evaluates the best over all segments (`_full_xseg`, `_null_xseg`). Evals: job 11447710.

Validation curves:
- **Adam (`_tigr`)** learns slowly. Best val N@10: plain .029 (step 88k), im .037 (78k),
  imc .038 (130k). Segment-local test: plain .0218, im .0267, imc .0267.
- **Adafactor (`_tigraf`)** peaks at step 4k, at the end of the first 4k steps of the constant
  lr 0.01 phase: val N@10 plain .038, im .042, imc .042. It then degrades to .027 to .033 by 200k.
  The segment-local test numbers (plain .0195, im .0221, imc .0233) are from the degraded late
  checkpoints and will be replaced by the cross-segment best.

### Is our plain TIGER too weak? (literature check 2026-11-05)

Plain TIGER test N@10, Beauty / Sports / Toys:

| source | Beauty | Sports | Toys | notes |
|---|---|---|---|---|
| TIGER paper (Rajput et al. 2023) | .0384 | .0225 | .0432 | no code released |
| GRID library, 5 runs (GBLA paper, SIGIR 2026, fork matfu-pixel/GRID) | .0355 | .0188 | .0333 | GRID's own TIGER |
| GRID paper (RK-Means row) | .0347 | | | |
| LIGER (Yang et al. 2024), own reimplementation + HP search, 3 seeds | .0322 | .0199 | .0295 | "TIGER's codebase unavailable" |
| community PyTorch ports (T-s-cell, NonameUntitled, phonism/genrec) | .0321 / .0319 / .0354 | | | |
| **ours `_es`** | .0327 | .0164 | .0306 | |
| **ours `_gp40rk`** | .0339 | .0188 | .0301 | |
| **ours `_tigr` (Adam)** | .0218 | | | far below everyone |

- `_es` and `_gp40rk` plain are inside the range of independent reproductions (Beauty .032 to .036).
  The TIGER paper's own numbers (Toys .0432 above all) are not matched by any reproduction found.
- Our weakest spots: `_es` Sports (.0164 vs .0188 to .0199) and Toys (.030 vs .033 in GRID's own runs).
- The stronger baseline does not shrink the gain: `_gp40rk` plain is higher than `_es` and im/imc gain more
  there (Beauty +19/+20% vs +13.5/+16%). im/imc on Beauty (.0401/.0405) are above every published plain
  TIGER number, including the paper's .0384.
- `_tigr` plain (.0218) is a poor reproduction (Adam at lr 0.01). Its +22% gain is measured against a weak
  baseline and should not be a headline.

## 4. What is running or queued

| setting | Beauty | Sports | Toys |
|---|---|---|---|
| `_es` | done | done | done |
| `_gp40rk` (replaces `_gprk`) | done | 5 done, 4 resuming | done |
| `_tigr` / `_tigraf` | dropped 2026-11-05 (pilot s2 only) | | |
| `_tigrtok` | parked | parked | parked |

Next results: TIGER pilot cross-segment evals, then the 4 resumed Sports `_gp40rk` runs.

## 5. Decisions 2026-11-05 (user)

1. **TIGER recipe: dropped (user, 2026-11-05).** First chosen Adam and released the 24 held chains; after the
   literature check (plain `_tigr` .0218, far below every reproduction) all `_tigr` / `_tigraf` jobs were
   cancelled (96 jobs incl. the pilot cross-segment evals). The comparison uses `_es` and `_gp40rk` only.
2. **Seeds 7, 11: run.** `_es` (4 arms) and `_gp40rk` (3 arms) on all three datasets: 42 trainings + evals,
   `logs/itemmem_seeds711_submit_20261105.tsv`. Not for `_tigr` (3 x 2-day chains each).
3. **Poisoned imc / plainscore controls: run.** The PKM paper grid (Beauty, 3 strata x data seeds 2,3,5,7,11),
   4,000-step training like plain / imsc, then pre + full F+R + NAU. 30 trainings + chains,
   `logs/grid_controls_submit_20261105.txt`. imc = memory + consistency, no score head (no reranking);
   plainscore = plain + score head, no input memory. Question: is imsc's gain the input memory or the reranking?

## 6. Results 2026-11-16: 5 seeds clean, poisoned architecture ablation

### Clean data, 5 seeds (scripts/itemmem_clean_summary.py)

| recipe | dataset | plain | im (gain, worst) | imc (gain, worst) | imfree (gain) |
|---|---|---|---|---|---|
| `_es` | Beauty | .0328 | .0376 (+14.7%, +11.8) | .0377 (+15.1%, +9.9) | .0324 (−1.0%) |
| `_es` | Sports | .0168 | .0177 (+5.9%, −11.7) | .0175 (+5.0%, −9.1) | .0130 (−22.2%) |
| `_es` | Toys | .0302 | .0377 (+24.9%, +13.4) | .0381 (+26.2%, +21.4) | .0321 (+6.4%) |
| `_gp40rk` | Beauty | .0337 | .0404 (+20.3%, +7.5) | .0404 (+20.1%, +13.0) | not run |
| `_gp40rk` | Sports | .0190 (4 seeds) | .0206 (+8.1%, −0.4; 4 seeds) | .0210 (+11.3%, +9.6; 3 seeds) | not run |
| `_gp40rk` | Toys | .0304 | .0381 (+25.3%, +20.7) | .0381 (+25.4%, +19.1) | not run |

Sports is the weak dataset: seeds 7, 11 pulled the `_es` gain from about +9% to +5 to 6%. Missing:
Sports `_gp40rk` imc s7, plain/im/imc s11 hit the 36 h limit (11447929, 11447939, 11447941,
11447943; evals cancelled). Not resumed: user asked on 2026-11-16 not to queue jobs (GPUs needed
for another project).

### Poisoned Beauty architecture ablation (scripts/itemmem_ablation_table.py)

15 cells (3 strata x 5 data seeds), 4000 steps. im and ims trained on the grid 2026-11-10
(logs/grid_imims_submit_20261110.txt). Test N@10 (paired gain vs TIGER) / SH@10:

| architecture | before | after F+R | after NAU |
|---|---|---|---|
| TIGER | .0313 / .0112 | .0319 / .0001 | .0319 / .0001 |
| + item memory (im) | .0360 (+15.2%) / .0097 | .0366 (+14.9%) / .0001 | .0365 (+14.9%) / .0001 |
| + memory + score (ims) | .0375 (+19.9%) / .0121 | .0380 (+19.2%) / .0001 | .0380 (+19.4%) / .0000 |
| + memory + score + consistency (imsc) | .0379 (+21.4%) / .0132 | .0381 (+19.5%) / .0001 | .0381 (+19.8%) / .0000 |
| + memory + consistency (imc) | .0358 (+14.6%) / .0131 | .0365 (+14.4%) / .0001 | .0365 (+14.8%) / .0001 |
| score only (plainscore) | .0337 (+7.9%) / .0133 | .0339 (+6.2%) / .0001 | .0340 (+6.8%) / .0000 |

### Pathways off at inference (OFF_EVALS=1, added 2026-11-10)

run_tiger_unlearn_sequential.sh OFF_EVALS=1 (default 1 in run_item_memory_beauty.sh) evaluates
the final model again into eval_off_{nulltoken,scoreoff,both}/. Memory contribution
(N_on − N_off)/N_on: im 29%, ims 32%, imc 26%, plainscore (score off) 3%; unchanged after
unlearning. Score off costs ~3% N@10 and never changes SH.

**Memory off triples SH before unlearning** (im .0095 → .0330, ims .0121 → .0377, imsc mid s2
.0120 → .0542): the spam is in the backbone and the memory suppresses it. After F+R / NAU, SH
stays ~0 with memory off (ims F+R .0004): unlearning removes it from the backbone, not only
behind the memory pathway.

Off evals missing for 30 models trained before OFF_EVALS existed (imsc 14 cells + all its
post-unlearning cells, imc 7, plainscore 6, im mid s2): rerun `STAGE=unlearn SPECS='<m>:pre
<m>:full'` when GPUs are free. Pilot 11497577/78 verified the on eval reproduces the old one.

### Update 2026-11-21: grid complete

- Off evals for the 30 older models done (88 reruns, logs/offeval_reruns_submit_20261120.txt):
  all 6 architectures now have on and off for all 15 cells, before / F+R / NAU. Memory off,
  N@10 before: im .0253, imc .0265, ims .0255, imsc .0270; SH@10 before: im .0343, imc .0365,
  ims .0377, imsc .0345; after F+R / NAU SH off <= .0004 everywhere.
- Consistency: no accuracy effect with memory on (imc vs im −0.4 to 0.0%, imsc vs ims +0.3 to
  +1.3%, within noise), but with memory off it keeps +5 to 6% more N@10 (imc > im in 10/15
  cells, imsc > ims in 11 to 14/15). Present it as robustness, not accuracy.
- Sports `_gp40rk` resumed (11644609/11/13/15), now 5 seeds: plain .0188, im .0208 (+10.7%,
  worst −0.4), imc .0203 (+8.2%, worst −8.8). On clean data im vs imc is a coin flip per seed
  on every dataset/recipe (mean differences −0.7 to +1.3%).
- Reruns re-evaluate imsc; newest run is used: imsc F+R .0380 (+19.4%), NAU .0382 (+20.1%).
