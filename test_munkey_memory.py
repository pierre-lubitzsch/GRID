"""CPU tests for the MUNKEY port (src/models/components/network_blocks/munkey_memory.py).
Run: python -m pytest -q test_munkey_memory.py"""

import torch
from torch.nn import functional as F

from src.models.components.network_blocks.munkey_memory import (
    MunkeyMemory,
    build_munkey_user_memory,
)


def _mem(**kw):
    torch.manual_seed(0)
    # 4 instances on two axes: 0,1 near e1; 2,3 near e2
    keys = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    return MunkeyMemory(keys, model_dim=6, token_dim=3, **kw)


def _raw(m, query, **kw):
    """retrieve() before the adapter: the weighted token mix itself."""
    w, b = m.adapter.weight, m.adapter.bias
    out = m.retrieve(query, **kw)
    return torch.linalg.lstsq(w, (out - b).T).solution.T


def test_retrieval_is_softmax_mix_of_nearest_and_masks():
    m = _mem(topk=2, tau=0.07).eval()
    q = torch.tensor([[1.0, 0.0]])
    keys = F.normalize(m.keys, dim=-1)
    sims = (F.normalize(q, dim=-1) @ keys.T)[0, :2]
    w = F.softmax(sims / 0.07, dim=-1)
    expect = w[0] * m.table.weight[0] + w[1] * m.table.weight[1]
    torch.testing.assert_close(_raw(m, q)[0], expect, rtol=1e-4, atol=1e-5)
    # own instance excluded -> the next two nearest (1 and 3)
    got = _raw(m, q, exclude_self=torch.tensor([0]))[0]
    sims2 = (F.normalize(q, dim=-1) @ keys.T)[0, [1, 3]]
    w2 = F.softmax(sims2 / 0.07, dim=-1)
    torch.testing.assert_close(got, w2[0] * m.table.weight[1] + w2[1] * m.table.weight[3],
                               rtol=1e-4, atol=1e-5)
    # a deleted key is never retrieved; exclude_self = -1 means no self
    assert m.delete(torch.tensor([1, 1, 9])) == 1 and m.deleted.tolist() == [False, True, False, False]
    got = _raw(m, q, exclude_self=torch.tensor([-1]))[0]
    sims3 = (F.normalize(q, dim=-1) @ keys.T)[0, [0, 3]]
    w3 = F.softmax(sims3 / 0.07, dim=-1)
    torch.testing.assert_close(got, w3[0] * m.table.weight[0] + w3[1] * m.table.weight[3],
                               rtol=1e-4, atol=1e-5)
    assert m.delete(torch.tensor([1])) == 0


def test_batch_exclusion_and_detached_neighbours_in_training():
    m = _mem(topk=2)
    m.train()
    q = m.keys[[0, 2]]
    out = m.retrieve(q, exclude=torch.tensor([0, 2]))
    out.sum().backward()
    assert m.table.weight.grad is None  # neighbours detached (freeze_memory)
    assert m.adapter.weight.grad is not None
    m.eval()
    m.zero_grad()
    m.retrieve(q).sum().backward()
    assert m.table.weight.grad.abs().sum() > 0  # eval: nothing detached
    m2 = _mem(topk=2, freeze_memory=False).train()
    m2.retrieve(q, exclude=torch.tensor([0, 2])).sum().backward()
    g = m2.table.weight.grad.abs().sum(1)
    assert g[0] == 0 and g[2] == 0 and g[1] > 0 and g[3] > 0  # batch excluded


def test_var_k_range_and_null_and_tokens():
    m = _mem(topk=4)
    ks = {m.sample_k() for _ in range(200)}
    assert ks == {2, 3, 4}
    assert _mem(topk=1).sample_k() == 1
    assert m.null_tokens(5).shape == (5, 6)
    torch.testing.assert_close(m.tokens(torch.tensor([2])), m.adapter(m.table.weight[[2]]))
    assert "deleted" in m.state_dict() and "keys" not in m.state_dict()


def test_build_from_key_file(tmp_path):
    f = tmp_path / "k.pt"
    torch.save({"user_ids": torch.tensor([3, 7, 10]), "neighbor_idx": torch.zeros(3, 1),
                "neighbor_sim": torch.zeros(3, 1), "keys": torch.randn(3, 5).half()}, f)
    m = build_munkey_user_memory(str(f), model_dim=8, token_dim=4)
    assert m.num_instances == 3 and m.uid_to_row.tolist()[3] == 0 and m.uid_to_row[7] == 1
    assert m.uid_to_row[10] == 2 and m.uid_to_row[5] == -1
    torch.save({"user_ids": torch.tensor([1])}, f)
    import pytest
    with pytest.raises(ValueError):
        build_munkey_user_memory(str(f), model_dim=8)


def test_munkey_delete_uses_full_forget_user_set():
    from types import SimpleNamespace as NS
    from torch import nn
    from src.components.unlearning.unified import _apply_munkey_delete

    model = nn.Module()
    model.munkey_memory = _mem()
    model.munkey_memory.register_buffer("uid_to_row", torch.tensor([-1, 0, 1, 2, 3]), persistent=False)
    batch = (NS(transformed_sequences={"user_id": torch.tensor([[1, 1], [2, 2]])}), None)
    # the sampled forget batch holds users 1, 2; the full request also has user 4
    info = _apply_munkey_delete(model, [batch], user_ids=torch.tensor([1, 2, 4, 9]))
    assert model.munkey_memory.deleted.tolist() == [True, True, False, True]
    assert info["munkey_keys_deleted_total"] == 3
    model.munkey_memory.deleted.zero_()
    _apply_munkey_delete(model, [batch])  # fallback: the batch's users only
    assert model.munkey_memory.deleted.tolist() == [True, True, False, False]
