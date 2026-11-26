"""Benchmark (read-only): generative L_sep scored per negative (current code) vs encode-once + chunked.

The current `_sep_loss_generative` runs one FULL encoder+decoder pass per negative item (and, with
sep_checkpoint, again in backward), although only the 4-token decoder target changes between
negatives. The candidate implementation below encodes each retain history once and scores the
negatives in chunks of K items by repeating the encoder output, optionally checkpointing each chunk.
Same objective; with dropout off (eval mode) loss and gradients must agree up to float rounding.

Nothing in src/ is modified and no run directory is written. Loads a trained model, takes one real
retain batch (batch size 32, the runs' unlearning batch size) of the retrained model's training data,
uses the deletion request's real category items as negatives, and reports loss / grad agreement
(eval mode), time of forward+backward (train mode, as in the runs) and peak memory.

    python -m scripts.bench_sep_encode_once experiment=tiger_unlearn_scif_flat data_dir=<retrain dir> \
        semantic_id_path=<sids> ckpt_path=<ckpt> num_hierarchies=4 +unlearning.bench_neg_json=<info.json>
"""
import json
import os
import time
from copy import deepcopy

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.components.unlearning.hvp import batch_to_device  # noqa: E402
from src.measure_retraining_js import _load_state  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import _build_finite_loader  # noqa: E402
from src.data.loading.components.collate_functions import collate_fn_train  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


def _decode_log_prob(model, enc, enc_mask, fut):
    """Sum over levels of log p(fut | history) given an already encoded history ([B'] from [B', T, D])."""
    out = model.decoder_forward_pass(future_ids=fut, attention_mask=None, encoder_output=enc,
                                     attention_mask_for_encoder=enc_mask, use_cache=False,
                                     past_key_values=None)[:, :-1]
    lp = out.new_zeros(fut.size(0))
    for h in range(model.num_hierarchies):
        log_p = torch.log_softmax(model.decoder.decoder_mlp[h](out[:, h]), dim=-1)
        lp = lp + log_p.gather(1, fut[:, h].long().unsqueeze(1)).squeeze(1)
    return lp


def sep_loss_generative_fast(model, retain_batch, neg_ids, temperature=1.0, chunk=32, checkpoint=True):
    """Encode once, score negatives in chunks of `chunk` items. Same loss as `_sep_loss_generative`."""
    model_input, label_data = retain_batch
    device = next(model.parameters()).device
    n_h = int(model.num_hierarchies)
    bsz = int(model_input.mask.size(0))
    fut_ids = None
    for key in label_data.labels:
        fut_ids = label_data.labels[key].reshape(bsz, -1)
    kw = {model.feature_to_model_input_map.get(k, k): v for k, v in model_input.transformed_sequences.items()}
    enc, enc_mask = model.encoder_forward_pass(attention_mask=model_input.mask, input_ids=kw["input_ids"],
                                               user_id=kw.get("user_id"))
    pos_lp = _decode_log_prob(model, enc, enc_mask, fut_ids[:, :n_h].long())          # [B]
    neg_sids = model.codebooks[torch.tensor(neg_ids)].to(device).long()               # [N, H]

    def score(e, m, sids):                                                              # -> [B, K]
        k = sids.size(0)
        e_r = e.unsqueeze(0).expand(k, *e.shape).reshape(k * bsz, *e.shape[1:])
        m_r = m.unsqueeze(0).expand(k, *m.shape).reshape(k * bsz, *m.shape[1:])
        f_r = sids.repeat_interleave(bsz, dim=0)                                      # item-major
        return _decode_log_prob(model, e_r, m_r, f_r).view(k, bsz).t()

    cols = []
    for j in range(0, neg_sids.size(0), chunk):
        s = neg_sids[j:j + chunk]
        if checkpoint and torch.is_grad_enabled():
            cols.append(torch.utils.checkpoint.checkpoint(score, enc, enc_mask, s, use_reentrant=False))
        else:
            cols.append(score(enc, enc_mask, s))
    logits = torch.cat([pos_lp.unsqueeze(1)] + cols, dim=1) / float(temperature)    # [B, 1+N]
    return (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()


def _run(model, fn, reps):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t = time.perf_counter()
        loss = fn(); loss.backward()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss), grads, times, torch.cuda.max_memory_allocated() / 2**30


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    info = json.load(open(str(cfg.unlearning.bench_neg_json)))
    neg_ids = list(info["requests"][0]["scif_info"]["target_items"])
    reps = int(cfg.unlearning.get("bench_reps", 3))
    with pipeline_launcher(cfg) as pipeline:
        model = pipeline.model
        _load_state(model, str(cfg.ckpt_path))
        device = torch.device("cuda")
        model = model.to(device)
        load_cfg = deepcopy(pipeline.datamodule.stage_to_config[TrainerFn.FITTING])
        load_cfg.collate_fn = collate_fn_train
        load_cfg.batch_size_per_device = 32
        batch = batch_to_device(next(iter(_build_finite_loader(
            load_cfg, os.path.join(str(cfg.paths.data_dir), "training")))), device)
        print(f"GPU {torch.cuda.get_device_name()}; negatives={len(neg_ids)}; batch={batch[0].mask.size(0)}", flush=True)

        old = lambda: model._sep_loss_generative(batch, neg_ids=neg_ids, temperature=1.0, checkpoint=True)
        variants = [("current (per-item full pass, sep_checkpoint)", old)]
        n_all = len(neg_ids)
        for k in (8, 32, n_all):
            variants.append((f"encode-once chunk={k} checkpoint", lambda k=k: sep_loss_generative_fast(
                model, batch, neg_ids, chunk=k, checkpoint=True)))
        variants.append((f"encode-once chunk={n_all} no checkpoint", lambda: sep_loss_generative_fast(
            model, batch, neg_ids, chunk=n_all, checkpoint=False)))

        # 1) agreement, dropout off
        model.eval()
        ref = None
        for name, fn in variants:
            loss, g, _, _ = _run(model, fn, 1)
            if ref is None:
                ref = (loss, g); print(f"[eval] {name}: loss {loss:.6f}", flush=True); continue
            d = max((ref[1][n] - g[n]).abs().max().item() for n in ref[1])
            sc = max(ref[1][n].abs().max().item() for n in ref[1])
            print(f"[eval] {name}: loss {loss:.6f} (ref {ref[0]:.6f}); max|grad diff| {d:.2e} of max|grad| {sc:.2e}; "
                  f"same params: {set(g) == set(ref[1])}", flush=True)
        # 2) timing, train mode as in the runs (one warm-up, then reps)
        model.train()
        for name, fn in variants:
            _run(model, fn, 1)
            loss, _, ts, mem = _run(model, fn, reps)
            print(f"[train] {name}: fwd+bwd {sum(ts) / len(ts):.3f} s (min {min(ts):.3f}); peak {mem:.1f} GiB", flush=True)
        retain = lambda: model._batch_loss_from_model_step(batch)
        _run(model, retain, 1)
        _, _, ts, _ = _run(model, retain, reps)
        print(f"[train] retain loss only (reference): fwd+bwd {sum(ts) / len(ts):.3f} s", flush=True)


if __name__ == "__main__":
    main()
