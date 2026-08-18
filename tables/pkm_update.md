# PKM as a Modular Unlearning Module — Status & Findings

*Extends the working notes with what has actually been measured on GRID/TIGER.
Everything marked **[measured]** has SLURM job ids and lives in `WORKFLOW.md`
(sections F–I); **[hypothesis]** is not yet tested. Unless stated otherwise all
results are **beauty, bandwagon attack, n_target=1, `mid` target, single seed**.*

Target: ICLR 2027, **Sep 25 2026 AOE**.

---

## 1. Motivation (unchanged)

**Transformer Feed-Forward Layers Are Key-Value Memories**
(https://aclanthology.org/2021.emnlp-main.446/)

- Transformers already store knowledge as "memory": FFNs behave like key–value
  memories storing internal facts.
- **Keys** capture patterns in the input; **values** encode what the model should
  output (a preference distribution over the output vocabulary).
- Each FFN output is a composition of multiple values; across layers FFNs refine
  predictions step by step.

**Takeaway.** Forgetting after fine-tuning is like overwriting internal memory →
we want a *modular stabilizer* instead of more backbone edits.

> **[measured] This framing is now empirically supported in a specific way.** In
> an encoder–decoder recommender the spam is **decoder-localised**: destroying and
> rebuilding any single *decoder* FFN removes 89–99% of spam exposure, while doing
> the same to any *encoder* FFN removes **nothing** (0 to −11%). Since FFN *values*
> encode output distributions, and generation happens decoder-side, this is exactly
> what Geva et al. predict. See §5.3.

### Prior work

1. Large memory layers with product keys — Lample et al., 2019
2. Large product key memory for pretrained language models — Kim et al., 2020
3. Memory layers at scale — Berges et al., 2024
4. Continual Learning via Sparse Memory Finetuning (SMF) — Lin et al., 2025

Related:
- Selective and Collaborative Influence Function for Efficient Recommendation
  Unlearning — https://arxiv.org/pdf/2304.10199
- Are we making progress in unlearning? NeurIPS unlearning competition —
  https://arxiv.org/pdf/2406.09073v1
- On privacy concerns of rehearsal learning — https://arxiv.org/pdf/2507.12305

---

## 2. The central question, and where it now stands

> *Instead of fine-tuning the whole model to remove spam, use a product-key memory
> head as a sparse correction module.*

Two variants were distinguished in the notes. Both have now been run.

| variant | what it is | status |
|---|---|---|
| **A. Trained-in** | PKM replaces FFNs during training; unlearning edits memory slots | **[measured]** works — matches full-model unlearning |
| **B. Post-hoc** | PKM inserted into an already-trained model, then adapted | **[measured]** works — and *dominates* every baseline |

### Answering the doc's own open question

> *"If PKM is added after training, it has no target information to REMOVE????"*

**[measured] Correct — and it does not need any.** The post-hoc variant works, but
**not** by removing target information from memory. Installing a PKM in `replace`
mode *discards* the trained FFN weights for that layer; the subsequent adaptation
sees **only retain data** (no forget term at all). So the mechanism is:

> **destroy the layer → rebuild it from clean data → the spam has no path back.**

This is a **localised retrain**, much closer to the retrain gold standard than to
seif/kookmin-style destroy-and-hope — but cheap: 1560 steps, one layer pair,
backbone frozen. It reframes the post-hoc branch: it is not "a corrective memory
that suppresses spam", it is "a cheap way to retrain the part of the network that
holds the spam".

---

## 3. Headline results

All on the same base attack (poison SH@10 0.01274 for the non-PKM model / 0.01248
for E23D01; non-PKM clean N@10 0.03763, clean SH@10 0.00000).

| method | SH@10 | removal | UR | note |
|---|---|---|---|---|
| poison baseline | 0.01274 | — | 0.962 | |
| `seif` | ~0 | ~full | **0.67–0.74** | utility floor, *not* noise-magnitude driven |
| `unified` ac0 (full model) | 0.00411 | 67% | 1.013 | best prior algorithm |
| **trained-in PKM, `pkm_only`** | **0.00385** | **69%** | **1.004** | backbone **frozen**, 0.2% utility cost |
| **post-hoc `d01` + repair** | **0.00013** | **99%** | **0.995** | **dominant**; clean-equivalent on both axes |

**[measured]** The post-hoc row dominates everything in the project's tables:
99% removal at essentially clean utility. The trained-in row is interesting for a
different reason — it needs **no layer destruction at all** and is the gentlest
update we have measured.

---

## 4. What we learned about PKM itself

### 4.1 Utility parity holds — placement matters more than capacity

**[measured]** Clean N@10 by architecture (non-PKM reference 0.03763):

| arch | clean N@10 | vs non-PKM |
|---|---|---|
| D0 | 0.03558 | −5.4% |
| D01 replace / add | 0.03645 / 0.03679 | −3.1% / −2.2% |
| Dall | 0.03398 | −9.7% |
| ALLv2 (enc+dec all) | 0.03596 | −4.4% |
| **E23D01** (enc tail + dec head) | **0.03835** | **+1.9%** |

Trained-in single decoder layers, poisoned (utility vs non-PKM poisoned 0.03619):

| model | N@10 | SH@10 | reading |
|---|---|---|---|
| D1only | **0.03607** | 0.02111 | utility **parity** (−0.3%), spam **1.66×** |
| D2only | 0.03508 | 0.00648 | utility −3.1%, spam **0.51×** |

**Not monotone in how many FFNs are replaced.** Placement dominates. Spam pickup
splits the *opposite* way from utility (D1 amplifies, D2 damps) — the same
non-monotonicity found in the architecture study.

> **Caveat that governs all of the above.** Three *identical-config* non-PKM clean
> runs span **7.5%** (0.03763 / 0.03615 / 0.03499, sd 3.6%). The clean-baseline
> seed noise is **larger** than the architecture differences. So: ALLv2 is
> statistically indistinguishable from non-PKM; "E23D01 beats non-PKM" is
> suggestive (its single run exceeds all three) but unreplicated. **No few-percent
> utility claim is safe without ≥3 seeds on both numerator and denominator.**

### 4.2 Separate architecture cost from unlearning cost

Dividing by a *non-PKM* clean baseline conflates two things. Against each model's
**own** clean baseline:

| arch | architecture cost | **unlearning cost** |
|---|---|---|
| ALLv2 | −4.4% | 0.9% |
| E23D01 | +1.9% | 0.5% |
| E23D01 `pkm_only` | +1.9% | **0.2%** |

**[measured]** PKM-model unlearning is *gentler* than the raw UR column suggests,
and `pkm_only` is the gentlest of all — which is precisely the modular-stabilizer
claim. Always report absolute nDCG next to UR and say which denominator.

### 4.3 !! The memory is collapsed !!

**[measured]** This is the most consequential finding and it changes priorities.

On the trained-in E23D01 model, with `n_keys=512` → **262,144 slots per memory**:

| memory | slots | forget reads | **slots touched** |
|---|---|---|---|
| enc block 2 | 262,144 | 9,600,000 | **32** |
| enc block 3 | 262,144 | 9,600,000 | **32** |
| dec block 0 | 262,144 | 320,000 | **96** |
| dec block 1 | 262,144 | 320,000 | **32** |

**0.012% of the memory is ever read.** The arithmetic is conclusive: `knn=32`,
`heads=4`, ~75,000 tokens, yet the union of every (token, head) top-32 is
*exactly* 32 slots, and reads divide perfectly evenly (9,600,000 / 32 = 300,000
each). **Every token and every head selects the identical 32 slots** — the
product-key routing is entirely input-independent and the heads have collapsed
onto each other.

This is the classic PKM pathology. Our config has `query_batchnorm=false` — the
very mechanism Lample et al. use to spread usage. It is off for a real reason
(BatchNorm running stats are corrupted by TIGER's variable padding), but the
price is total collapse.

**Consequences**

1. **Top-t slot selection cannot be evaluated on these models.** Not because the
   criteria are wrong — there is nothing to select among. `forget_exclusive_slots`
   = 0, `touched_jaccard` = 1.000, and AF-IHF ≡ AF because IHF is *constant* over
   the 32 used slots.
2. **Explains the ~0.004 `pkm_only` ceiling**: effective capacity is ~32 × 128 =
   4096 params per memory, not 33.5M.
3. **Likely explains the architecture utility cost**: a dense, input-dependent FFN
   replaced by something returning the same 32 vectors for every token.
4. **Every PKM result in this project was obtained at 0.012% utilisation.** They
   are valid measurements, but they test a *low-rank replacement layer*, not the
   "large sparse memory" hypothesis.

### 4.4 Collapse is training-mode-dependent — post-hoc memory is healthy

**[measured]** The same diagnostic on the **repaired post-hoc** PKM (PKM inserted
over a trained model, then 1560 steps of PKM-only retain finetuning, backbone
frozen):

| model | slots | forget touched | retain touched | Jaccard | forget-exclusive |
|---|---|---|---|---|---|
| trained-in E23D01 | 262,144 | **32** | **32** | **1.000** | **0** |
| post-hoc repaired (dec 0) | 262,144 | 64,052 | **154,803** | **0.386** | **3,051** |
| post-hoc repaired (dec 1) | 262,144 | 74,827 | **182,920** | **0.382** | **3,550** |

Retain uses **59%** of the memory, forget and retain route **differently**, and
**~3,000 slots are forget-read and retain-untouched**. That is exactly the
structure top-t selection needs — and it exists **only in the post-hoc branch**.

**[hypothesis] Why.** The post-hoc PKM was trained *in isolation on a frozen
backbone*, so the query projection had no alternative but to learn input-dependent
routing. In joint training the rest of the network can route *around* the memory,
so it degenerates into a constant bias. If this holds, **"train the memory in
isolation" is a cheaper collapse fix than fighting `query_batchnorm`.**

**Consequences.** (i) The {100, 1000, 10000} t-grid *is* meaningful here; the
small grid applied only to the collapsed model. (ii) The repaired model has a
healthy memory but **no spam left** (SH 0.00013), so nothing to select *for*. The
configuration giving both is post-hoc **`add`** mode — PKM alongside the FFN,
backbone frozen, warmed on retain → healthy memory with spam still in the frozen
backbone → then selective editing. That is precisely the *post-hoc correction
version* in the notes.

**This answers another of the doc's questions** — *"How long does it take for the
PKM query/value structure to have a meaningful representation?"* On our runs
(~6.5h training, full convergence): **it never did.**

**Before any more PKM experiments: fix utilisation.** Candidate causes to
separate — (a) `query_batchnorm=false` (needs a padding-safe alternative: LayerNorm
on the query, or BatchNorm over non-pad tokens only); (b) `query_proj` collapsing
to a near-constant map; (c) keys never moving from their uniform init. (b) and (c)
are cheap read-only checks on an existing checkpoint. **Slot utilisation is now a
first-class metric, not an afterthought.**

---

## 5. Post-hoc branch in detail

### 5.1 The data-path bug that nearly buried it

First attempt: utility *degraded* monotonically with more repair steps
(UR 0.832 → 0.569 → 0.360). Cause was **not** under-training —
`retain_samples_used_for_update=16` caps the retain data, so the fine-tune saw
~2 batches out of 22,363 available rows and simply memorised them. Loss went
9.84 → 1.99 and plateaued while test utility collapsed.

Fix: `unlearning.retain_source = subset (default) | full`. With the full retain
split (~88 batches), `d01` converges to **UR 0.995**. Non-breaking — verified by
re-running two configs on defaults and reproducing recorded numbers exactly.

> **Lesson worth keeping.** Loss-plateau early stopping **cannot** see this failure:
> loss on a memorised sample looks healthy while generalisation dies. `dall`
> early-stopped at 5784 steps with a healthy `best_loss=1.79` while utility had
> already fallen 0.891 → 0.686. **Any repair loop needs a held-out utility check,
> not a loss plateau.**

### 5.2 A destruction threshold

| destroyed | 2000 steps | 10000 steps | behaviour |
|---|---|---|---|
| `d01` (2 FFNs) | UR 0.995 | UR 0.995 | **converges** (early stop 1560, identical) |
| `dall` (4 FFNs) | UR 0.891 | UR 0.686 | **no stable solution**, degrades with steps |

Destroy 2 FFNs and the rebuild converges to clean-equivalence; destroy 4 and there
is no stable solution even with the full retain set.

### 5.3 Single-layer placement screen — the spam is decoder-localised

Post-hoc PKM at **one** layer at a time, non-PKM poisoned base:

| layer | SH@10 | removed | UR |
|---|---|---|---|
| **d0** | 0.00139 | 89.1% | **0.978** |
| **d1** | 0.00031 | 97.5% | 0.958 |
| d2 | 0.00009 | **99.3%** | 0.904 |
| d3 | 0.00054 | 95.8% | 0.898 |
| e0 | 0.01306 | −2.5% | 0.978 |
| e1 | 0.01413 | −10.9% | 0.978 |
| e2 | 0.01288 | −1.1% | 0.960 |
| e3 | 0.01400 | −9.9% | 0.944 |

**Every encoder layer removes nothing.** All four decoder layers remove 89–99%.
Corroborated independently by the RQ-ID gradient diagnosis (forget gradients
concentrated on the per-code decoder heads).

Within the decoder, **deeper removes more but costs more utility**. And `d01`
(both, 99% / **0.995**) beats every single layer — so the optimum is the *first two
decoder FFNs together*, not fewer. These placement differences are deterministic
for this base model (the pipeline reproduces exactly), but transfer across
strategies and base-model seeds is untested.

---

## 6. Top-t slot selection

### 6.1 Which history for IHF

**Retain**, for two reasons: it is definitionally what must be preserved (using the
full training stream would fold forget data into IHF exactly where it must
discriminate), and the counts are **additive**, so they can be accumulated once and
updated incrementally as new retain data arrives. Use `retain_source=full` — the
`16×|D_f|` subset understates HF.

### 6.2 Criteria implemented

`unlearning.slot_selection = none | af | af_ihf | grad | grad_combined`

| criterion | score | note |
|---|---|---|
| `af` | forget access frequency | access-count baseline (SMF's TF term) |
| `af_ihf` | `AF(s) · log((T_r+1)/(HF(s)+1))` | **AF-IHF**, our TF-IDF analogue |
| `grad` | `‖g_f‖ − λ‖g_r‖` | magnitudes only |
| `grad_combined` | `ĝf − λ·ĝr − μ·dot` | `dot_i = ⟨g_f,i, g_r,i⟩`, all max-normalised |

**Why AF-IHF rather than TF-IDF.** TF-IDF's IDF is `log(N_docs / df)` — a *document*
count, which needs a notion of documents. SMF has one (pretraining corpus batches);
a recommender does not. AF-IHF replaces document frequency with **retain read
volume**, measurable from the same counters, with no arbitrary batch-as-document
choice.

**Why the dot term is the right collateral-damage measure.** The unlearning update
moves along `+g_f`, so the first-order change in the **retain** loss from editing
slot *i* is `⟨g_f,i, g_r,i⟩` — **not** `‖g_r,i‖`. It is *signed*: negative means
editing for forgetting also *improves* retain, which a norm-based score cannot
express.

> **[measured] Verified on a synthetic case.** Two slots with identical
> `‖g_f‖ = ‖g_r‖ = 12` but `dot` of 0 vs +144 rank **1007 vs 1006** under
> `g_f − g_r` (indistinguishable), and **1007 vs 1999 (last)** under the combined
> score. Only the inner product separates them.

**Useful property.** Restricted to the selected coordinates,
`g_f,I^T g_r,I = Σ_{i∈I} ⟨g_f,i, g_r,i⟩` is **additively separable over slots**, so
exact top-t is plain `topk` — no greedy approximation. (This breaks only if the
absolute value goes *outside* the sum.)

### 6.3 Choosing t

The notes propose t ∈ {25, 50, 100, 200, 500, 1000}. **That grid is right for a
healthy memory and meaningless for a collapsed one**: with 32 live slots, t=25
already selects 78% of the usable memory, and any t ≥ 32 is a *no-op* because dead
slots have exactly zero gradient — masking them in or out changes nothing.

`select_top_t_slots` therefore **logs a warning when `t ≥ live slots`**, so a
selection that isn't restricting anything cannot masquerade as a result.

Currently running: t ∈ {1, 2, 4, 8, 16, 32, 100, 1000, 10000} × {`af_ihf`,
`grad_combined`} on E23D01. Prediction on record: everything from t=32 up should be
**identical**, and `af_ihf` should be indistinguishable from plain `af`.

---

## 7. Optimizer

SMF uses **SGD**, not Adam. The argument (from the notes) is sound: Adam's moment
estimates get diluted on steps where a sparsely-selected slot receives zero
gradient, distorting effective step sizes; SGD has no cross-step state to corrupt.

**[measured] In our sweep Adam beat SGD at every comparable setting.** This is
**not** a refutation: that sweep updated **all** memory params, which is exactly the
regime where SMF's argument does not apply. **Re-test once top-t selection is
active** — that is the setting the argument is about.

Both knobs exist: `unlearning.unified_optimizer` and `finetune_optimizer`
(`adam` | `sgd`, SGD with momentum 0.9).

---

## 8. Implementation status

| capability | knob | status |
|---|---|---|
| PKM in training | `PKM_MODE`, `PKM_ENCODER`, `PKM_DECODER` | done |
| PKM post-hoc insertion | same + `strict=False` ckpt load | done (works today) |
| Memory-only update | `unlearning.update_scope=pkm_only` | done (unified + finetune) |
| Value-table-only update | `pkm_update_keys/query` | done |
| Full retain stream | `unlearning.retain_source=full` | done |
| Early stopping | `finetune_patience`, `finetune_min_delta` | done (loss-based; see §5.1 caveat) |
| SGD | `*_optimizer=sgd` | done |
| Slot access counters | `HashingMemory.enable_access_counting()` | done (`persistent=False`, schema-safe) |
| Slot diagnostics | `src/diagnose_pkm_slots.py` | done |
| Top-t selection | `slot_selection`, `slot_top_t`, `slot_lambda`, `slot_mu` | done |
| Single-checkpoint retention | `checkpoint_fractions: [1.0]` | done (halved disk) |

**Not built:** zero value-init for post-hoc `add` mode (values init
`normal_(0, v_dim**-0.5)`, so an added PKM perturbs the residual stream from step 0
— a zeros init would make it exactly identity and give a true corrective adapter);
held-out utility check inside the repair loop; any fix for memory collapse.

---

## 9. Revised plan

**Blocking everything:**

1. **Fix memory collapse.** Diagnose (query variance / keys-vs-init — both cheap and
   read-only), then try a padding-safe query normalisation. Re-measure slot
   utilisation as an acceptance criterion. *Until this is done, the entire
   selection line is untestable and every PKM number describes a low-rank layer.*

**Then, in order:**

2. Re-run the top-t sweep at the notes' intended scale (t ∈ 100…10000) with
   AF-IHF vs the gradient criterion, plus the access-count baseline as the
   ablation, and Adam vs SGD *within* selection.
3. Explain the shared ~0.004 ceiling on the trained-in branch. Both full-model and
   `pkm_only` stop there, and it is **not** a metric floor (clean SH is 0.00004) —
   and post-hoc reaches 0.00013 on the same attack, so it is not irreducible either.
4. Post-hoc `add` + zero-init: the non-destructive counterpart to the replace
   results, and the true "modular stabilizer" configuration.
5. **Seeds.** Everything above is single-seed, `mid`-only, beauty-only. Cheapest
   high-value confirmation: replicate post-hoc `d01` across all three target
   strategies (minutes per run). Exclude the ALLv2-`popular` cell — the attack
   never landed there (poison SH 0.00018 vs clean 0.00013).

**Open, not yet investigated:** whether robustness/attack analysis for generative
retrieval is genuinely unexplored (determines how the contribution is positioned);
the LSTM-gating connection from the 31.07 meeting notes.

---

## 10. Spam taxonomy — coverage

| spam type | example | our coverage |
|---|---|---|
| Fake users | bot accounts promoting target items | **bandwagon**, clone_flood, clone_inject |
| Fake interactions | artificial clicks/ratings/reviews | same generators |
| Spam items | low-popularity items pushed into popular sessions | **`segment`**, `clone_append` |
| Coordinated campaigns | many fake users promoting items/categories | `n_target` > 1 sweeps |

Nearly complete — a taxonomy table with a method per row is cheap to write and
strengthens the robustness-analysis angle the notes flag as an open question.

---

## Appendix A — pipeline

The standard GenRec/GenIR unlearning pipeline, with the memory idea entering at
step 4:

1. Train model
2. Identify target to forget
3. Build forget and retain data
4. **Apply unlearning update → update only selected memory slots**
5. Evaluate forget effectiveness and retain utility

Idealised memory pipeline (from the notes, with status):

1. Train GenRec with PKM, or add PKM and warm it on clean data — **done, both**
2. Define forget set — done
3. Define retain set — done
4. Observe which memory slots are used / receive gradients — **done** (§4.3; the
   answer was "only 32")
5. Select editable slots: matters for target, not for retain — **implemented,
   untestable until collapse is fixed**
6. Freeze everything else (backbone, SID codebook, keys, most values) —
   **done** (`update_scope=pkm_only`, `pkm_update_keys/query`)
7. Update only selected memory values — **done** (row-masked gradients)
8. Evaluate: target exposure down, nDCG stable — done

## Appendix B — PKM sizing

Given hidden size `d_model`:

```
value_dim   = d_model
key_dim     = d_model / 2      (we currently use k_dim = d_model)
num_values  = as large as the memory budget allows
num_subkeys = sqrt(num_values)
```

Current TIGER config: `k_dim=128 (=d_model)`, `heads=4`, `knn=32`, `n_keys=512`
→ 262,144 slots. **Given §4.3, `n_keys` is not the binding constraint — routing is.
Raising capacity without fixing collapse buys nothing.**

## Appendix C — key file map

| what | where |
|---|---|
| PKM core + access counters | `src/models/components/network_blocks/product_key_memory.py` |
| Layer wrappers, install logic | `src/models/modules/semantic_id/tiger_generation_model.py` |
| Slot selection criteria | `src/components/unlearning/slot_selection.py` |
| Memory-only param scope | `src/components/unlearning/target_params.py::select_pkm_params` |
| Unlearning dispatch | `src/models/modules/semantic_id/tiger_unlearning_module.py` |
| Slot diagnostics | `src/diagnose_pkm_slots.py`, `run_diagnose_pkm_slots.sh` |
| Full experiment record | `WORKFLOW.md` sections F–I |
