"""CPU tests for the item-keyed exemplar memory and its unlearning policy.
Run: python -m pytest -q test_item_memory.py"""

import torch
from torch import nn

from src.components.unlearning.unified import _apply_item_memory_policy
from src.models.components.network_blocks.item_memory import ItemExemplarMemory, nearest_items


def _mem(p_drop=0.0, p_nbr=0.0, n=6, d=4):
    torch.manual_seed(0)
    emb = torch.randn(n, 8)
    idx, sim = nearest_items(emb, 3)
    return ItemExemplarMemory(n, d, idx, sim, p_drop=p_drop, p_nbr=p_nbr), emb


def test_nearest_items_excludes_self_and_is_sorted():
    _, emb = _mem()
    idx, sim = nearest_items(emb, 3, chunk=2)
    assert not (idx == torch.arange(6).unsqueeze(1)).any()
    assert (sim[:, :-1] >= sim[:, 1:]).all()


def test_eval_uses_own_entry_and_masks_padding():
    m, _ = _mem(p_drop=0.5, p_nbr=0.4)
    m.eval()
    ids = torch.tensor([[0, 3, -1]])
    out = m(ids)
    torch.testing.assert_close(out[0, 1], m.adapter(m.values.weight[3]))
    assert (out[0, 2] == 0).all()


def test_training_substitutes_neighbours_without_grad_and_drops_to_null():
    m, _ = _mem(p_drop=0.3, p_nbr=0.5)
    m.train()
    with torch.no_grad():
        m.null_value.fill_(5.0)
    ids = torch.zeros(1, 4000, dtype=torch.long)  # item 0 everywhere
    out = m(ids)
    null_out = m.adapter(m.null_value)
    own_out = m.adapter(m.values.weight[0])
    is_null = torch.isclose(out[0], null_out).all(-1)
    is_own = torch.isclose(out[0], own_out).all(-1)
    frac_null, frac_own = is_null.float().mean().item(), is_own.float().mean().item()
    assert 0.25 < frac_null < 0.35 and 0.15 < frac_own < 0.25  # rest = neighbours
    out.sum().backward()
    g = m.values.weight.grad
    nbrs = set(m.neighbor_idx[0].tolist())
    assert g[0].abs().sum() > 0
    assert all(g[j].abs().sum() == 0 for j in nbrs)  # neighbours get no gradient


def test_access_counting():
    m, _ = _mem()
    m.enable_access_counting()
    m(torch.tensor([[1, 1, 2, -1]]))
    c = m.get_access_counts()
    assert c[1] == 2 and c[2] == 1 and c.sum() == 3


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_memory, _ = _mem()
        self.backbone = nn.Linear(4, 1)

    def model_step(self, ids):
        return None, self.backbone(self.item_memory(ids)).sum()


def test_policy_lr_scales_and_afihf_rows():
    model = _Toy()
    params = [p for p in model.parameters() if p.requires_grad]
    mult = {id(p): 1.0 for p in params}
    masks = {}
    forget = [(torch.tensor([[1, 2, 2]]),)]
    retain = [(torch.tensor([[2, 2, 2, 3, 3]]),)]
    info = _apply_item_memory_policy(
        model, params, mult, masks, forget, retain,
        lr_scale=10.0, backbone_scale=0.0, rows="afihf", top_t=1,
    )
    vid = id(model.item_memory.values.weight)
    assert mult[vid] == 10.0
    assert mult[id(model.backbone.weight)] == 0.0
    # item 1: read by forget only -> highest AF-IHF; item 2 is mostly retain
    assert masks[vid].squeeze(1).nonzero().flatten().tolist() == [1]
    assert info["n_items_forget"] == 2 and info["n_rows_selected"] == 1


def test_policy_without_item_memory_is_noop_or_rejects():
    model = nn.Linear(2, 2)
    p = list(model.parameters())
    assert _apply_item_memory_policy(model, p, {id(x): 1.0 for x in p}, {}, [], [],
                                     lr_scale=1.0, backbone_scale=1.0, rows="all", top_t=1) == {}
    try:
        _apply_item_memory_policy(model, p, {id(x): 1.0 for x in p}, {}, [], [],
                                  lr_scale=5.0, backbone_scale=1.0, rows="all", top_t=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---- output-side score, soft rows, reset, local repair rows ------------------------
def test_item_score_uses_own_entry_and_zero_for_padding():
    m, _ = _mem()
    m.enable_item_score()
    h = torch.randn(2, 4)
    ids = torch.tensor([[0, 5, -1], [3, 3, 1]])
    s = m.item_score(h, ids)
    assert s.shape == (2, 3) and s[0, 2] == 0
    torch.testing.assert_close(s[1, 0], (h[1] * m.out_proj(m.values.weight[3])).sum())


def test_policy_soft_rows_and_reset():
    model = _Toy()
    params = [p for p in model.parameters() if p.requires_grad]
    mult = {id(p): 1.0 for p in params}
    masks = {}
    w = model.item_memory.values.weight
    before = w.detach().clone()
    forget = [(torch.tensor([[1, 2, 2]]),)]
    retain = [(torch.tensor([[2, 2, 2, 3, 3]]),)]
    info = _apply_item_memory_policy(model, params, mult, masks, forget, retain,
                                     lr_scale=1.0, backbone_scale=1.0, rows="afihf_soft",
                                     top_t=0, reset_alpha=0.5)
    scale = info["_row_scale"]
    assert scale[1] == 1.0 and 0 < scale[2] < 1 and scale[0] == 0  # 2 is retain-heavy
    assert masks[id(w)].squeeze(1).nonzero().flatten().tolist() == [1, 2]
    assert info["n_rows_reset"] == 2
    assert torch.equal(w[0], before[0]) and not torch.equal(w[1], before[1])


def test_local_row_masks():
    from src.components.unlearning.unified import _local_row_masks

    class M:
        num_hierarchies = 2
        _adaptive_sorted_keys = torch.zeros(1)

        def _codes_to_item_ids(self, codes):  # item id = first code, -1 if negative
            return torch.where(codes[..., 0] >= 0, codes[..., 0], torch.full_like(codes[..., 0], -1))

    class MI:
        def __init__(self, seq):
            self.transformed_sequences = {"sequence_data": seq}

    seq = torch.tensor([[3, 0, 5, 0], [1, 0, -1, -1], [5, 9, 7, 0]])
    aff = torch.zeros(10, dtype=torch.bool)
    aff[5] = True
    (m,) = _local_row_masks(M(), [(MI(seq), None)], aff)
    assert m.tolist() == [True, False, True]


def test_force_neighbor_mode_is_deterministic_and_detached():
    m, _ = _mem(p_drop=0.5, p_nbr=0.4)
    m.train()
    m.force_mode = "neighbor"
    ids = torch.tensor([[0, 2]])
    a, b = m(ids), m(ids)
    torch.testing.assert_close(a, b)
    a.sum().backward()
    assert m.values.weight.grad is None or m.values.weight.grad.abs().sum() == 0
    assert m.adapter.weight.grad.abs().sum() > 0


def test_pkm_item_policy_masks_blocks():
    from src.components.unlearning.unified import _apply_pkm_item_policy
    from src.models.components.network_blocks.product_key_memory import HashingMemory

    class MI:
        def __init__(self, seq):
            self.transformed_sequences = {"sequence_data": seq}

    class M(nn.Module):
        num_hierarchies = 1

        def __init__(self):
            super().__init__()
            self.mem = HashingMemory(input_dim=8, output_dim=8, k_dim=4, heads=1, knn=2,
                                     n_keys=4, item_slots=2, num_items=4)
            self.codebooks = torch.zeros(4, 1)
            self._adaptive_sorted_keys = torch.zeros(1)

        def _ensure_item_id_map(self):
            pass

        def _codes_to_item_ids(self, codes):
            return codes[..., 0]

    m = M()
    params = list(m.parameters())
    mult = {id(p): 1.0 for p in params}
    masks = {}
    forget = [(MI(torch.tensor([[1, 2]])), None)]
    retain = [(MI(torch.tensor([[2, 2, 3]])), None)]
    info = _apply_pkm_item_policy(m, params, mult, masks, forget, retain,
                                  rows="afihf", top_t=1, lr_scale=5.0)
    assert mult[id(m.mem.item_values.weight)] == 5.0 and mult[id(m.mem.values.weight)] == 1.0
    assert masks[id(m.mem.item_values.weight)].flatten().tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert info["n_items_selected"] == 1


# ---- user-keyed exemplar memory ------------------------------------------------------
def _user_file(tmp_path):
    uids = torch.tensor([3, 7, 10, 42])
    nbr = torch.tensor([[1, 2], [0, 2], [3, 0], [2, 1]])
    sim = torch.rand(4, 2)
    f = tmp_path / "nbrs.pt"
    torch.save({"user_ids": uids, "neighbor_idx": nbr, "neighbor_sim": sim}, f)
    return str(f)


def test_user_memory_lut_and_unknown_user(tmp_path):
    from src.models.components.network_blocks.item_memory import build_user_memory
    um = build_user_memory(_user_file(tmp_path), dim=4)
    assert um.uid_to_row[7] == 1 and um.uid_to_row[42] == 3 and um.uid_to_row[5] == -1
    um.eval()
    out = um(torch.tensor([um.uid_to_row[10], -1]))
    assert (out[1] == 0).all() and out[0].abs().sum() > 0


def test_user_policy_rows_reset_and_zero_shot_rows(tmp_path):
    from src.components.unlearning.unified import _apply_user_memory_policy
    from src.models.components.network_blocks.item_memory import build_user_memory

    class MI:
        def __init__(self, u):
            self.transformed_sequences = {"user_id": u}

    m = nn.Module()
    m.user_memory = build_user_memory(_user_file(tmp_path), dim=4)
    w = m.user_memory.values.weight
    before = w.detach().clone()
    params = list(m.parameters())
    mult = {id(p): 1.0 for p in params}
    masks = {}
    forget = [(MI(torch.tensor([[7, 0], [42, 0], [5, 0]])), None)]  # 5 unknown
    info = _apply_user_memory_policy(m, params, mult, masks, forget,
                                     lr_scale=10.0, rows="forget", reset_alpha=0.0)
    assert mult[id(w)] == 10.0 and info["n_forget_users"] == 2
    assert masks[id(w)].flatten().tolist() == [0, 1, 0, 1]
    assert torch.equal(w[0], before[0]) and not torch.equal(w[1], before[1])
    # alpha 0 -> exactly the weighted neighbour average (rows 0, 2 for user 7)
    wts = torch.softmax(m.user_memory.neighbor_sim[1] / m.user_memory.tau, dim=-1)
    torch.testing.assert_close(w[1], wts[0] * before[0] + wts[1] * before[2])


def test_backbone_scale_with_user_memory_only(tmp_path):
    from src.components.unlearning.unified import _apply_item_memory_policy
    from src.models.components.network_blocks.item_memory import build_user_memory
    m = nn.Module()
    m.user_memory = build_user_memory(_user_file(tmp_path), dim=4)
    m.backbone = nn.Linear(4, 4)
    params = list(m.parameters())
    mult = {id(p): 1.0 for p in params}
    _apply_item_memory_policy(m, params, mult, {}, [], [], lr_scale=1.0, backbone_scale=0.0,
                              rows="all", top_t=1)
    assert mult[id(m.user_memory.values.weight)] == 1.0
    assert mult[id(m.backbone.weight)] == 0.0 and mult[id(m.user_memory.adapter.weight)] == 0.0


def test_explicit_drop_mask_applies_in_eval_and_excludes_sampling(tmp_path):
    from src.models.components.network_blocks.item_memory import build_user_memory
    um = build_user_memory(_user_file(tmp_path), dim=4, p_drop=0.9, p_nbr=0.0)
    with torch.no_grad():
        um.null_value.fill_(3.0)
    rows = torch.tensor([0, 1, 2, 3])
    null_out = um.adapter(um.null_value)
    um.train()  # p_drop=0.9 must NOT be sampled when a mask is given
    out = um(rows, drop_mask=torch.tensor([True, False, False, False]))
    assert torch.isclose(out[0], null_out).all()
    assert not torch.isclose(out[1:], null_out).all(-1).any()
    um.eval()  # forced drop also works at eval (pathway sensitivity)
    out = um(rows, drop_mask=torch.tensor([False, True, False, False]))
    assert torch.isclose(out[1], null_out).all() and not torch.isclose(out[0], null_out).all()


def test_local_batches_from_full_pool():
    import dataclasses
    from src.components.unlearning.unified import _local_batches_from_pool, gather_batch_rows

    @dataclasses.dataclass
    class MI:
        transformed_sequences: dict
        mask: torch.Tensor

    @dataclasses.dataclass
    class LD:
        labels: torch.Tensor   # flattened n x H
        powers: torch.Tensor   # per-hierarchy, not row-major

    class M:
        num_hierarchies = 2
        _adaptive_sorted_keys = torch.zeros(1)

        def _codes_to_item_ids(self, codes):
            return torch.where(codes[..., 0] >= 0, codes[..., 0], torch.full_like(codes[..., 0], -1))

    def batch(seq):
        n = seq.size(0)
        return (MI({"sequence_data": seq}, torch.arange(n)),
                LD(torch.arange(n * 2) + 100 * seq[:, 0].repeat_interleave(2), torch.tensor([1, 7])))

    # pool: 2 batches; affected item 5 is in rows (0,0), (0,2), (1,1)
    pool = [batch(torch.tensor([[5, 0, 1, 0], [2, 0, 3, 0], [4, 0, 5, 0]])),
            batch(torch.tensor([[1, 0, 2, 0], [5, 0, 6, 0]]))]
    aff = torch.zeros(10, dtype=torch.bool)
    aff[5] = True

    g = gather_batch_rows([(pool[0], torch.tensor([2, 0]))])
    assert g[0].transformed_sequences["sequence_data"].tolist() == [[4, 0, 5, 0], [5, 0, 1, 0]]
    assert g[0].mask.tolist() == [2, 0] and g[1].powers.tolist() == [1, 7]
    assert g[1].labels.tolist() == [404, 405, 500, 501]  # per-row label pairs follow their rows
    # a 1-row part must not turn per-hierarchy fields into row-major ones
    g = gather_batch_rows([(pool[0], torch.tensor([1])), (pool[1], torch.tensor([0]))])
    assert g[1].powers.tolist() == [1, 7] and g[0].mask.tolist() == [1, 0]

    (lb,), hits = _local_batches_from_pool(M(), pool, aff, max_rows=10, seed=0)
    assert hits == 3
    seqs = lb[0].transformed_sequences["sequence_data"]
    assert seqs.size(0) == 3 and lb[1].labels.numel() == 6 and lb[1].powers.tolist() == [1, 7]
    assert all(5 in r.view(-1, 2)[:, 0].tolist() for r in seqs)

    (lb,), hits = _local_batches_from_pool(M(), pool, aff, max_rows=2, seed=0)
    assert hits == 3 and lb[0].transformed_sequences["sequence_data"].size(0) == 2
    assert _local_batches_from_pool(M(), pool, torch.zeros(10, dtype=torch.bool), 4, 0) == ([], 0)


# ---- recipe="munkey" (the authors' training / inference recipe) -----------------
def _mk(p_drop=0.0, p_nbr=0.0, tau=0.07):
    torch.manual_seed(0)
    emb = torch.randn(6, 8)
    idx, sim = nearest_items(emb, 4)
    return ItemExemplarMemory(6, 4, idx, sim, p_drop=p_drop, p_nbr=p_nbr, tau=tau,
                              recipe="munkey")


def test_munkey_recipe_eval_retrieves_self_plus_neighbours():
    m = _mk()
    m.eval()
    out = m(torch.tensor([[2, -1]]))
    sim = torch.cat([torch.ones(1), m.neighbor_sim[2, :3]])
    nid = torch.cat([torch.tensor([2]), m.neighbor_idx[2, :3]])
    w = torch.softmax(sim / 0.07, dim=-1)
    expect = m.adapter((m.values.weight[nid] * w.unsqueeze(-1)).sum(0))
    torch.testing.assert_close(out[0, 0], expect)
    assert (out[0, 1] == 0).all()
    assert not torch.allclose(out[0, 0], m.adapter(m.values.weight[2]))  # not the own entry alone


def test_munkey_recipe_training_draws_per_batch():
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    m = _mk(p_nbr=1.0)  # every batch: neighbour mix, detached
    m.train()
    m(ids).sum().backward()
    assert m.values.weight.grad is None or m.values.weight.grad.abs().sum() == 0
    m = _mk()  # no draw fires: own entries, with gradient
    m.train()
    m(ids).sum().backward()
    assert (m.values.weight.grad.abs().sum(1) > 0).all()
    m = _mk(p_drop=0.999)  # the whole batch nulled at once
    m.train()
    with torch.no_grad():
        m.null_value.fill_(3.0)
    out = m(ids)
    torch.testing.assert_close(out, m.adapter(torch.full((4,), 3.0)).expand_as(out))


def test_munkey_recipe_validation():
    import pytest
    emb = torch.randn(4, 8)
    idx, sim = nearest_items(emb, 2)
    with pytest.raises(ValueError):
        ItemExemplarMemory(4, 4, idx, sim, recipe="other")
    ItemExemplarMemory(4, 4, idx, sim, p_drop=0.2, p_nbr=1.0, recipe="munkey")  # p_nbr may be 1


def test_item_deletion_substitutes_neighbours_everywhere_and_loads_old_ckpts():
    m, _ = _mem()
    m.eval()
    m.enable_item_score()
    ids = torch.tensor([[2, 4]])
    before = m(ids).detach().clone()
    s_before = m.item_score(torch.ones(1, 4), torch.tensor([[2]])).detach().clone()
    assert m.delete([2, 2, 99]) == 1 and m.deleted.tolist()[2]
    after = m(ids)
    nid, sim = m.neighbor_idx[2], m.neighbor_sim[2]
    w = torch.softmax(sim / m.tau, -1)
    torch.testing.assert_close(after[0, 0], m.adapter((m.values.weight[nid] * w.unsqueeze(-1)).sum(0)))
    torch.testing.assert_close(after[0, 1], before[0, 1])      # other items unchanged
    assert not torch.allclose(m.item_score(torch.ones(1, 4), torch.tensor([[2]])), s_before)
    # a deleted neighbour is masked out of another item's substitution
    other = int(nid[0]); m.delete([other])
    k = (m.neighbor_idx[2] != other)
    w2 = torch.softmax(sim[k] / m.tau, -1)
    torch.testing.assert_close(m(ids)[0, 0], m.adapter((m.values.weight[nid[k]] * w2.unsqueeze(-1)).sum(0)))
    # checkpoints from before the buffer existed still load strictly
    sd = {k_: v for k_, v in m.state_dict().items() if k_ != "deleted"}
    m2, _ = _mem(); m2.enable_item_score()
    m2.load_state_dict(sd, strict=True)
    assert not m2.deleted.any()


def test_item_deletion_masks_munkey_retrieval():
    m = _mk(); m.eval()
    out0 = m(torch.tensor([[2]]))
    m.delete([2])
    sim = m.neighbor_sim[2, :3]; nid = m.neighbor_idx[2, :3]
    w = torch.softmax(sim / 0.07, -1)
    torch.testing.assert_close(m(torch.tensor([[2]]))[0, 0], m.adapter((m.values.weight[nid] * w.unsqueeze(-1)).sum(0)))
    assert not torch.allclose(out0, m(torch.tensor([[2]])))


def test_policy_noop_when_item_memory_outside_restricted_scope():
    """update_scope=pkm_only leaves item_memory.values out of the update set: fine with
    default item knobs (the 11295282-90 failures), an error when an item knob is set."""
    m, _ = _mem()
    model = nn.Module(); model.item_memory = m; model.other = nn.Linear(2, 2)
    p = list(model.other.parameters())                    # values NOT in the update set
    mult = {id(x): 1.0 for x in p}
    assert _apply_item_memory_policy(model, p, mult, {}, [], [], lr_scale=1.0,
                                     backbone_scale=1.0, rows="all", top_t=1) == {}
    import pytest
    with pytest.raises(ValueError):
        _apply_item_memory_policy(model, p, mult, {}, [], [], lr_scale=10.0,
                                  backbone_scale=1.0, rows="all", top_t=1)


def test_item_drift_projection_caps_rows_relative_to_anchor():
    from src.components.unlearning.unified import _item_drift_stats, _project_item_drift
    anchor = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])     # norms 5, 1, 2
    w = anchor.clone()
    w[0] += torch.tensor([30.0, 40.0])     # drift 50 = 10x its norm -> capped at 0.5 x 5
    w[1] += torch.tensor([0.1, 0.0])       # drift 0.1 = 0.1x -> untouched
    _project_item_drift(w, anchor, 0.5)
    rel = (w - anchor).norm(dim=1) / anchor.norm(dim=1)
    assert torch.allclose(rel, torch.tensor([0.5, 0.1, 0.0]), atol=1e-6)
    assert torch.allclose((w[0] - anchor[0]) / (w[0] - anchor[0]).norm(),
                          torch.tensor([0.6, 0.8]))          # direction kept
    st = _item_drift_stats(w, anchor, 0.5)
    assert st["rows_moved"] == 2 and st["rows_at_cap"] == 1 and abs(st["rel_drift_max"] - 0.5) < 1e-6


def test_item_drift_projection_row_mask_caps_only_selected_rows():
    from src.components.unlearning.unified import _item_drift_stats, _project_item_drift
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    w = anchor + torch.tensor([[2.0, 0.0], [2.0, 0.0]])      # both drift 2x their norm
    rows = torch.tensor([True, False])
    _project_item_drift(w, anchor, 0.5, rows=rows)
    assert torch.allclose(w, torch.tensor([[1.5, 0.0], [3.0, 0.0]]))  # row 1 left free
    st = _item_drift_stats(w, anchor, 0.5, rows=rows)
    assert st["rows_moved"] == 1 and st["rows_at_cap"] == 1


def test_group_memory_reads_only_its_group_and_masks_padding():
    from src.models.components.network_blocks.item_memory import GroupMemory
    torch.manual_seed(0)
    gm = GroupMemory(num_groups=4, dim=6, slots=5, knn=2, key_dim=3)
    gm.eval()
    groups = torch.tensor([[1, 2, -1]])
    emb = torch.randn(1, 3, 6)
    out = gm(groups, emb)
    assert (out[0, 2] == 0).all()                                   # padding
    out[0, :2].sum().backward()
    gv = gm.values.grad.abs().sum(-1)                               # (groups, slots)
    assert set(gv.sum(-1).nonzero().flatten().tolist()) == {1, 2}   # only the read groups
    assert int((gv[1] > 0).sum()) == 2 and int((gv[2] > 0).sum()) == 2   # knn slots each
    gm.zero_grad(); gm.train()
    gm(groups, emb).sum().backward()
    assert gm.null_value.grad is not None                           # in the graph (DDP)
