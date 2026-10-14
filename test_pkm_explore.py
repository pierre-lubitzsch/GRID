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
