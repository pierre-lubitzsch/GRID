"""CPU tests for the PKM explore / exploit mode (HashingMemory.set_explore and
PKMExploreController). Run: python -m pytest -q test_pkm_explore.py"""

import torch
from torch.nn import functional as F

from src.models.components.network_blocks.product_key_memory import (
    HashingMemory,
    PKMExploreController,
)


def _mem(**kw):
    torch.manual_seed(0)
    return HashingMemory(input_dim=16, output_dim=16, k_dim=8, heads=2, knn=4, n_keys=8, **kw)


def _brute_dense(mem, x):
    """Softmax over ALL n_keys**2 pair scores, built explicitly."""
    q = mem.query_proj(x).view(-1, mem.heads, mem.k_dim)
    half = mem.k_dim // 2
    out = 0
    for h in range(mem.heads):
        s1 = q[:, h, :half] @ mem.keys[h, 0].T
        s2 = q[:, h, half:] @ mem.keys[h, 1].T
        s = (s1[:, :, None] + s2[:, None, :]).reshape(len(x), -1)  # slot i*n+j
        out = out + F.softmax(s, -1) @ mem.values.weight
    return out


def test_dense_read_matches_brute_force():
    mem = _mem()
    mem.train()
    x = torch.randn(5, 16)
    mem.set_explore(True, forward="dense", chunk=2)
    torch.testing.assert_close(mem(x), _brute_dense(mem, x), rtol=1e-5, atol=1e-6)


def test_dense_step_updates_all_slots():
    mem = _mem()
    mem.train()
    mem.set_explore(True, forward="dense")
    mem(torch.randn(3, 16)).sum().backward()
    assert (mem.values.weight.grad.abs().sum(1) > 0).all()
    assert (mem.keys.grad.abs().sum(-1) > 0).all()


def test_straight_through_keeps_sparse_forward():
    mem = _mem()
    mem.train()
    x = torch.randn(4, 16)
    sparse = mem(x)
    mem.set_explore(True, forward="straight_through")
    out = mem(x)
    torch.testing.assert_close(out, sparse)
    out.sum().backward()
    assert (mem.values.weight.grad.abs().sum(1) > 0).all()


def test_exploit_and_eval_are_sparse():
    mem = _mem()
    mem.set_explore(True, forward="dense")
    mem.eval()  # explore is ignored outside training
    mem(torch.randn(3, 16)).sum().backward()
    rows = (mem.values.weight.grad.abs().sum(1) > 0).sum().item()
    assert rows <= 3 * mem.heads * mem.knn


def test_util_tracking_only_on_exploit_passes():
    mem = _mem()
    mem.train()
    mem.enable_util_tracking()
    mem.set_explore(True, forward="dense")
    mem(torch.randn(3, 16))
    assert mem.pop_util_seen().sum() == 0
    mem.set_explore(False)
    mem(torch.randn(3, 16))
    seen = mem.pop_util_seen()
    assert 0 < seen.sum() <= 3 * mem.heads * mem.knn
    assert mem.pop_util_seen().sum() == 0  # reset


def test_controller_schedules_and_determinism():
    c = PKMExploreController(p=0.3, schedule="constant", seed=1)
    draws = [c.draw(s)[0] for s in range(2000)]
    assert draws == [c.draw(s)[0] for s in range(2000)]
    assert 0.25 < sum(draws) / len(draws) < 0.35

    lin = PKMExploreController(p=0.5, p_final=0.1, schedule="linear", start_step=100, decay_steps=100)
    assert lin.prob(0) == 0.5 and abs(lin.prob(150) - 0.3) < 1e-9 and abs(lin.prob(10_000) - 0.1) < 1e-9
    cos = PKMExploreController(p=0.5, p_final=0.1, schedule="cosine", decay_steps=100)
    assert abs(cos.prob(50) - 0.3) < 1e-9

    ad = PKMExploreController(p=0.4, p_final=0.05, schedule="adaptive", util_target=0.5)
    assert ad.prob(0) == 0.4
    ad.util = 0.25
    assert abs(ad.prob(0) - 0.225) < 1e-9
    ad.util = 0.9
    assert ad.prob(0) == 0.05
    assert PKMExploreController(p=0.0).draw(7) == (False, 0.0)


# ---- routing balance loss -----------------------------------------------------
import math


def test_balance_loss_bounds_and_degenerate_case():
    mem = _mem()
    mem.train()
    n = mem.n_keys
    flat = torch.zeros(6, n)  # every token: identical, flat routing (collapse-compatible)
    mem.set_balance(True, token_entropy_weight=0.0)
    mem._bal_h_marg, mem._bal_h_tok = [], []
    assert abs(mem._balance_term(flat).item()) < 1e-5  # marginal-only loss is fooled
    mem.set_balance(True, token_entropy_weight=1.0)
    assert abs(mem._balance_term(flat).item() - math.log(n)) < 1e-5  # MI loss is not
    onehot = torch.full((n, n), -1e4)
    onehot[torch.arange(n), torch.arange(n)] = 0.0  # each token its own sub-key
    assert abs(mem._balance_term(onehot).item()) < 1e-4  # perfect balance -> 0


def test_balance_loss_is_set_and_trains_router():
    mem = _mem()
    mem.train()
    mem.set_balance(True)
    mem(torch.randn(5, 16))
    assert mem.balance_loss is not None and mem.balance_loss.item() > 0
    assert set(mem.balance_stats) == {"eff_subkeys_batch", "eff_subkeys_token"}
    mem.balance_loss.backward()
    assert mem.keys.grad.abs().sum() > 0
    assert mem.query_proj[0].weight.grad.abs().sum() > 0
    mem.eval()
    mem(torch.randn(5, 16))
    assert mem.balance_loss is None  # training only


def _switch(mem, scores):
    mem._bal_h_marg, mem._bal_h_tok = [], []
    return mem._balance_term(scores)


def test_switch_balance_values_scale_invariance_and_gradient():
    mem = _mem()  # n_keys=8, knn=4
    mem.train()
    mem.set_balance(True, kind="switch")
    torch.manual_seed(0)
    # collapsed: every token ranks sub-keys 0..3 on top
    collapsed = torch.randn(64, 8) * 0.1
    collapsed[:, :4] += 3.0
    # spread: each token prefers a different half
    spread = torch.randn(64, 8) * 0.1
    spread[::2, :4] += 3.0
    spread[1::2, 4:] += 3.0
    lc, ls = _switch(mem, collapsed).item(), _switch(mem, spread).item()
    assert lc > ls and abs(ls - 1.0) < 0.1  # uniform hard routing -> ~1
    # scale invariance: shrinking every score must not lower the loss
    assert abs(_switch(mem, collapsed * 1e-3).item() - lc) < 1e-3
    # functional: a router that starts collapsed, trained on this loss alone,
    # spreads its HARD top-k assignments
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    lin = torch.nn.Linear(16, 8)
    with torch.no_grad():
        lin.bias.copy_(torch.tensor([3.0] * 4 + [0.0] * 4))  # sub-keys 0-3 win for everyone
    def eff_assigned():
        _switch(mem, lin(x))
        return mem._bal_h_tok[-1].exp().item()
    before = eff_assigned()
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        _switch(mem, lin(x)).backward()
        opt.step()
    after = eff_assigned()
    assert before < 4.2 and after > 7.0  # 4 of 8 sub-keys used -> nearly all 8


def test_switch_balance_sets_assigned_stat():
    mem = _mem()
    mem.train()
    mem.set_balance(True, kind="switch")
    mem(torch.randn(5, 16))
    assert set(mem.balance_stats) == {"eff_subkeys_batch", "eff_subkeys_assigned"}
    assert mem.balance_loss.item() >= 1.0 - 1e-4


# ---- memory (pathway) dropout ----------------------------------------------------
def test_memory_dropout_replaces_whole_sequences_in_training_only():
    torch.manual_seed(0)
    mem = HashingMemory(input_dim=16, output_dim=16, k_dim=8, heads=2, knn=4, n_keys=8,
                        memory_dropout=0.5)
    with torch.no_grad():
        mem.null_value.fill_(7.0)
    x = torch.randn(64, 5, 16)  # 64 sequences of 5 tokens
    mem.train()
    out = mem(x)
    dropped = (out == 7.0).all(-1)  # (64, 5)
    assert (dropped.all(1) | (~dropped).all(1)).all()  # per sequence, never per token
    assert 10 < dropped.all(1).sum().item() < 54
    out.sum().backward()
    assert mem.null_value.grad.abs().sum() > 0  # the null vector is learned
    mem.eval()
    assert not (mem(x) == 7.0).all(-1).any()  # inactive at eval


def test_memory_dropout_off_adds_no_param():
    mem = HashingMemory(input_dim=16, output_dim=16, k_dim=8, heads=2, knn=4, n_keys=8)
    assert "null_value" not in mem.state_dict()


# ---- item-addressed slots ---------------------------------------------------------
def _imem():
    torch.manual_seed(0)
    return HashingMemory(input_dim=16, output_dim=16, k_dim=8, heads=2, knn=4, n_keys=8,
                         item_slots=3, num_items=5)


def test_item_slots_read_only_own_block():
    mem = _imem()
    mem.train()
    x = torch.randn(1, 4, 16)
    base = mem(x).detach()  # no context -> shared read only
    mem.set_item_context(torch.tensor([[2, 2, -1, 4]]))
    out = mem(x)
    assert torch.equal(out[0, 2], base[0, 2])  # token without item unchanged
    assert not torch.equal(out[0, 0], base[0, 0])
    out.sum().backward()
    g = mem.item_values.weight.grad.abs().sum(1)
    touched = set(g.nonzero().flatten().tolist())
    assert touched == {6, 7, 8, 12, 13, 14}  # blocks of items 2 and 4 only
    gk = mem.item_keys.grad.abs().sum((1, 2))
    assert set(gk.nonzero().flatten().tolist()) == {2, 4}


def test_item_slots_skip_on_layout_mismatch():
    mem = _imem()
    mem.set_item_context(torch.tensor([[1, 2]]))  # 2 ids for 4 tokens
    assert mem._item_read(torch.randn(4 * 2, 8), 4) is None


def test_switch_raw_scores_are_scale_dependent():
    mem = _mem()
    mem.train()
    mem.set_balance(True, kind="switch", standardize=False)
    torch.manual_seed(0)
    collapsed = torch.randn(64, 8) * 0.1
    collapsed[:, :4] += 3.0
    # without standardisation, shrinking all scores flattens P and lowers the
    # loss towards 1 although the top-k assignment is still collapsed
    assert _switch(mem, collapsed * 1e-3).item() < _switch(mem, collapsed).item() - 0.5


def test_slot_selection_protects_top_retain_slots(monkeypatch):
    import src.components.unlearning.slot_selection as ss
    mem = _mem()  # 64 slots
    af = torch.zeros(64, dtype=torch.float64); af[[1, 2, 3, 4]] = torch.tensor([9.0, 8.0, 7.0, 6.0], dtype=torch.float64)
    hf = torch.zeros(64, dtype=torch.float64); hf[[1, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = 1.0; hf[1] = 100.0
    sweeps = iter([{"m": af}, {"m": hf}])
    monkeypatch.setattr(ss, "_memories", lambda model: [("m", mem)])
    monkeypatch.setattr(ss, "_access_sweep", lambda model, mems, batches: next(sweeps))
    masks, info = ss.select_top_t_slots(None, [], [], criterion="af", top_t=2, protect_frac=0.1)
    sel = masks["m"].flatten().nonzero().flatten().tolist()
    assert 1 not in sel and sel == [2, 3]  # slot 1 = most retain-used -> protected
    assert info["per_memory"]["m"]["protected_slots"] == 1


# ---- query_norm from the checkpoint (auto / adopt / mismatch guard) ----------------
def _wrapped(query_norm):
    torch.manual_seed(0)
    w = torch.nn.Module()
    w.mem = HashingMemory(input_dim=16, output_dim=16, k_dim=8, heads=2, knn=4, n_keys=8,
                          query_norm=query_norm)
    return w


def test_auto_query_norm_resolves_from_checkpoint():
    import pytest
    from src.models.components.network_blocks.product_key_memory import (
        check_query_norm_load, resolve_auto_query_norm)

    bn_sd = _wrapped("batchnorm").state_dict()
    plain_sd = _wrapped("none").state_dict()
    assert any(".q_norm." in k for k in bn_sd) and not any(".q_norm." in k for k in plain_sd)

    w = _wrapped("auto")
    assert resolve_auto_query_norm(w, bn_sd) == [("mem", "batchnorm")]
    w.load_state_dict(bn_sd)  # strict: keys match after resolution
    w = _wrapped("auto")
    assert resolve_auto_query_norm(w, plain_sd) == [("mem", "none")]
    w.load_state_dict(plain_sd)
    w = _wrapped("auto")  # fresh post-hoc PKM: not in the checkpoint at all
    assert resolve_auto_query_norm(w, {}) == [("mem", "none")]

    # explicit settings are left alone, and a mismatch is an error
    w = _wrapped("none")
    assert resolve_auto_query_norm(w, bn_sd) == []
    with pytest.raises(ValueError):
        check_query_norm_load(w, bn_sd)
    w = _wrapped("batchnorm")
    with pytest.raises(ValueError):
        check_query_norm_load(w, plain_sd)
    check_query_norm_load(w, {})  # untrained post-hoc memory: fine

    # adopt (evaluation): the checkpoint wins over an explicit setting
    w = _wrapped("batchnorm")
    assert resolve_auto_query_norm(w, plain_sd, adopt=True) == [("mem", "none")]
    w.load_state_dict(plain_sd)
    w = _wrapped("none")
    assert resolve_auto_query_norm(w, bn_sd, adopt=True) == [("mem", "batchnorm")]
    w.load_state_dict(bn_sd)


def test_balance_reduction_sum_is_per_memory():
    """reduction=sum gives every memory the full coefficient (= coef * n under mean)."""
    from types import SimpleNamespace as NS
    import src.utils  # noqa: F401  (import order as in src.train; avoids a circular import)
    from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder
    mems = [("a", NS(balance_loss=torch.tensor(2.0), balance_stats={})),
            ("b", NS(balance_loss=torch.tensor(4.0), balance_stats={}))]
    fake = NS(_pkm_memories=lambda: mems, log=lambda *a, **k: None, device="cpu")
    fake._pkm_balance_reduction = "mean"
    assert float(SemanticIDEncoderDecoder._pkm_balance_loss(fake)) == 3.0
    fake._pkm_balance_reduction = "sum"
    assert float(SemanticIDEncoderDecoder._pkm_balance_loss(fake)) == 6.0
    del fake._pkm_balance_reduction                # configs without the key keep mean
    assert float(SemanticIDEncoderDecoder._pkm_balance_loss(fake)) == 3.0


def test_slot_mask_excludes_masked_slots_from_reads():
    mem = _mem()  # 64 slots, knn 4
    mem.eval()
    x = torch.randn(6, 16)
    q = mem.query_proj(x).view(-1, mem.heads, mem.k_dim)
    _, idx = mem._get_indices(q[:, 0], mem.keys[0])
    read = idx.unique()
    mem.slot_mask = torch.zeros(mem.size, dtype=torch.bool)
    mem.slot_mask[read] = True                      # delete every slot read before
    _, idx2 = mem._get_indices(q[:, 0], mem.keys[0])
    assert not torch.isin(idx2, read).any()
    mem.slot_mask = None
    _, idx3 = mem._get_indices(q[:, 0], mem.keys[0])
    assert torch.equal(idx3, idx)                   # no mask -> unchanged


def test_ffn_pathway_dropout_in_add_mode():
    import src.utils  # noqa: F401  (import order)
    from transformers import T5Config
    from transformers.models.t5.modeling_t5 import T5LayerFF
    from src.models.modules.semantic_id.tiger_generation_model import T5LayerFFWithPKM, T5LayerPKM
    cfg = T5Config(d_model=16, d_ff=32, dropout_rate=0.0)
    pp = dict(k_dim=8, heads=2, knn=4, n_keys=8)
    torch.manual_seed(0)
    layer = T5LayerFFWithPKM(T5LayerFF(cfg), cfg, {**pp, "ffn_dropout": 0.999})
    h = torch.randn(4, 5, 16)
    layer.train()
    mem = layer.memory(layer.layer_norm(h))
    torch.testing.assert_close(layer(h), h + mem)       # FFN branch skipped -> residual + memory
    layer.eval()
    torch.testing.assert_close(layer(h), layer.ffn(h) + layer.memory(layer.layer_norm(h)))
    import pytest
    with pytest.raises(ValueError):
        T5LayerPKM(cfg, {**pp, "ffn_dropout": 0.5})     # replace mode has no FFN to drop
