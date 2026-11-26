"""CPU unit test (tiny mock model, no checkpoint, no data) for the two rebuttal
variants of the unified objective, both OFF by default:

  * reference-capped forgetting (unlearning.ref_cap_forget / ref_cap_sep):
    forget term = hinge on target rows, separation = target negative leaves the
    contrast once below its cap;
  * retain-side neighbourhood anchor (unlearning.lambda_a).

Run:  python test_ref_cap_anchor.py
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch import nn

import src.utils  # noqa: F401  (import order of the real entry points; avoids a circular import)
import src.components.unlearning.unified as U
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder

# The mock batches are plain objects, not the TIGER dataclasses.
U.batch_to_device = lambda batch, device: batch
U.batch_size = lambda batch: int(batch[0].mask.shape[0])

H, K, D, N_ITEMS = 3, 5, 6, 12


class MockModel(nn.Module):
    """Teacher-forced sequence log-prob over H levels of K codes, conditioned on
    a history vector (stored in `mask`) and the previous code; has dropout so the
    train/eval distinction is real."""

    num_hierarchies = H

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.enc = nn.Linear(D, 8)
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleList([nn.Linear(8 + K, K) for _ in range(H)])
        self.register_buffer("codebooks", torch.randint(0, K, (N_ITEMS, H), generator=g))

    def _teacher_forced_log_prob(self, model_input, future_ids):
        h = self.drop(torch.tanh(self.enc(model_input.mask)))
        prev = torch.zeros(h.size(0), K)
        lp = h.new_zeros(h.size(0))
        for lvl in range(H):
            logp = torch.log_softmax(self.heads[lvl](torch.cat([h, prev], -1)), -1)
            tgt = future_ids[:, lvl].long()
            lp = lp + logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            prev = nn.functional.one_hot(tgt, K).float()
        return lp

    def _batch_loss_from_model_step(self, batch):
        mi, ld = batch
        return -self._teacher_forced_log_prob(mi, ld.labels["y"]).mean()

    # the REAL separation loss code
    compute_sep_loss = SemanticIDEncoderDecoder.compute_sep_loss
    _sep_loss_generative = SemanticIDEncoderDecoder._sep_loss_generative


def batches(model, n, b, seed, target=None, frac=0.5):
    out = []
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        items = torch.randint(0, N_ITEMS, (b,), generator=g)
        if target is not None:
            items[: int(b * frac)] = target
        mi = SimpleNamespace(mask=torch.randn(b, D, generator=g), transformed_sequences={})
        ld = SimpleNamespace(labels={"y": model.codebooks[items].clone()})
        out.append((mi, ld))
    return out


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def test_forget_hinge():
    lp = torch.tensor([-1.0, -3.0, -0.5, -4.0], requires_grad=True)
    nan = float("nan")
    v0, _, _ = U.capped_forget_log_prob(lp, torch.full((4,), nan))
    ok(torch.allclose(v0, lp.mean()), "no capped row -> plain mean log-prob")
    cap = torch.tensor([-2.0, -2.0, nan, -2.0])  # rows 0,1,3 are target rows
    v, n_c, n_a = U.capped_forget_log_prob(lp, cap)
    v.backward()
    ok(n_c == 3 and n_a == 1, f"3 capped rows, 1 above its cap ({n_c}, {n_a})")
    ok(torch.allclose(lp.grad, torch.tensor([0.25, 0.0, 0.25, 0.0])),
       f"gradient 1/B on active rows, 0 below the cap: {lp.grad.tolist()}")
    ok(abs(float(v) - (-1.0 - 0.5) / 4) < 1e-6, "value keeps the 1/B normaliser")


def test_sep_floor():
    torch.manual_seed(0)
    model = MockModel().eval()
    rb = batches(model, 1, 6, seed=1)[0]
    neg = {3}
    base = model.compute_sep_loss(rb, negative_item_ids=neg, loss_type="generative")
    lo = model.compute_sep_loss(rb, negative_item_ids=neg, loss_type="generative",
                                neg_log_prob_floor={3: torch.full((6,), -1e9)})
    ok(torch.allclose(base, lo), "floor below every log-prob -> identical L_sep")
    stats = {}
    hi = model.compute_sep_loss(rb, negative_item_ids=neg, loss_type="generative",
                                neg_log_prob_floor={3: torch.full((6,), 1e9)},
                                floor_stats=stats)
    ok(float(hi) == 0.0, "floor above every log-prob -> L_sep = 0")
    hi.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    ok(g == 0.0 and stats["sep_capped_active"] == 0, "... and zero gradient")
    # Mixed: the negative's gradient vanishes exactly on the rows below the floor.
    neg_sids = model.codebooks[3].unsqueeze(0).expand(6, H)
    lp_neg = model._teacher_forced_log_prob(rb[0], neg_sids).detach()
    floor = lp_neg.clone()
    floor[:3] = lp_neg[:3] + 1.0   # rows 0-2 below the floor
    floor[3:] = lp_neg[3:] - 1.0   # rows 3-5 above
    mi = rb[0]
    x = mi.mask.clone().requires_grad_(True)
    rb2 = (SimpleNamespace(mask=x, transformed_sequences={}), rb[1])
    l = model.compute_sep_loss(rb2, negative_item_ids=neg, loss_type="generative",
                               neg_log_prob_floor={3: floor})
    l.backward()
    # rows 0-2: only the positive's -log softmax(pos) with no negative = 0 -> grad 0
    ok(float(x.grad[:3].abs().sum()) == 0.0 and float(x.grad[3:].abs().sum()) > 0,
       "per-row: no gradient where the target is below its cap, gradient above it")
    try:
        model.compute_sep_loss(rb, negative_item_ids=neg, loss_type="cosine",
                               neg_log_prob_floor={3: floor})
        ok(False, "cosine + floor must raise")
    except ValueError:
        ok(True, "cosine separation refuses a floor")


def test_anchor_zero_at_reference():
    torch.manual_seed(0)
    model = MockModel().train()
    rb = batches(model, 1, 8, seed=2)[0]
    a_sids = model.codebooks[[1, 4, 7]]
    model.eval()
    with torch.no_grad():
        ref = torch.logsumexp(U.item_log_probs(model, rb[0], a_sids), 1)
    for ck in (False, True):
        model.zero_grad(set_to_none=True)
        lm = torch.logsumexp(U.item_log_probs(model, rb[0], a_sids, checkpoint=ck), 1)
        la = U.anchor_loss(lm, ref)
        la.backward()
        g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        ok(float(la) == 0.0 and g == 0.0,
           f"anchor = 0 with zero gradient at the reference weights (checkpoint={ck})")
    # Perturb, then a few SGD steps on the anchor alone bring the mass back.
    with torch.no_grad():
        model.heads[0].bias.add_(torch.tensor([2.0, -1.0, 0.5, 0.0, -2.0]))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    vals = []
    for _ in range(30):
        opt.zero_grad()
        lm = torch.logsumexp(U.item_log_probs(model, rb[0], a_sids, checkpoint=True), 1)
        la = U.anchor_loss(lm, ref)
        la.backward()
        opt.step()
        vals.append(float(la))
    ok(vals[0] > 0 and vals[-1] < 0.5 * vals[0], f"anchor descends {vals[0]:.4f} -> {vals[-1]:.4f}")
    model.train()


def _run(model, fb, rb, **kw):
    torch.manual_seed(123)
    base = dict(n_epochs=2, lr=1e-2, lambda_retain=1.0, lambda_forget=0.1,
                lambda_sep=0.01, sep_loss_type="generative", sep_negatives_mode="forget_target_only",
                sep_negative_item_ids={3}, forget_item_ids={3}, device=torch.device("cpu"))
    base.update(kw)
    return U.unified_unlearn(model, fb, rb, **base)


def test_end_to_end():
    torch.manual_seed(0)
    m0 = MockModel()
    fb = batches(m0, 2, 8, seed=3, target=3)
    rb = batches(m0, 3, 8, seed=4)
    sd = copy.deepcopy(m0.state_dict())

    def fresh():
        m = MockModel()
        m.load_state_dict(sd)
        return m

    a, b = fresh(), fresh()
    _run(a, fb, rb)
    nan = float("nan")
    lab_is_t = [(f[1].labels["y"] == a.codebooks[3]).all(-1) for f in fb]
    _run(b, fb, rb,
         forget_cap=[torch.where(t, torch.full((8,), -1e9), torch.full((8,), nan)) for t in lab_is_t],
         sep_cap=[{3: torch.full((8,), -1e9)} for _ in rb])
    same = all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), b.parameters()))
    ok(same, "caps that never bind reproduce the uncapped run")

    # All target rows capped from the start (+inf caps), forget term only: the
    # update must equal plain ascent on the NON-target rows alone.
    c = fresh()
    info_c = _run(c, fb, rb, lambda_retain=0.0, lambda_sep=0.0,
                  forget_cap=[torch.where(t, torch.full((8,), 1e9), torch.full((8,), nan)) for t in lab_is_t])
    st = info_c["ref_cap_stats"]
    ok(st["forget_capped_rows"] > 0 and st["forget_capped_active"] == 0,
       f"target rows fully capped: {st}")
    d = fresh()
    d.eval()
    lp = d._teacher_forced_log_prob(fb[0][0], fb[0][1].labels["y"])
    v, _, _ = U.capped_forget_log_prob(
        lp, torch.where(lab_is_t[0], torch.full((8,), 1e9), torch.full((8,), nan)))
    v.backward()
    keep = ~lab_is_t[0]
    lp2 = d._teacher_forced_log_prob(fb[0][0], fb[0][1].labels["y"])
    ref = (lp2 * keep.float()).sum() / 8
    ok(torch.allclose(v, ref.detach()), "capped forget value = non-target rows only (1/B)")

    # Anchor wired end to end: runs, reports, and is 0 on the first step.
    e = fresh()
    e.eval()
    a_sids = e.codebooks[[1, 4, 7]]
    with torch.no_grad():
        refs = [torch.logsumexp(U.item_log_probs(e, r[0], a_sids), 1) for r in rb]
    info_e = _run(e, fb, rb, lambda_anchor=1.0, anchor_sids=a_sids, anchor_ref=refs)
    ok(info_e["anchor_enabled"] and info_e["anchor_n_items"] == 3
       and info_e["mean_anchor_loss"] is not None, "anchor term runs inside unified_unlearn")
    ok(e.training, "model is back in train mode after the anchor's eval-mode passes")

    # Default call: new fields present and inert.
    f_ = fresh()
    info_f = _run(f_, fb, rb)
    ok(info_f["ref_cap_stats"] is None and not info_f["anchor_enabled"], "defaults: both off")


if __name__ == "__main__":
    test_forget_hinge()
    test_sep_floor()
    test_anchor_zero_at_reference()
    test_end_to_end()
    print("ALL PASSED")
