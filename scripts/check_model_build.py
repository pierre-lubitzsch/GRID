"""CPU smoke check: build the full TIGER model from the Hydra config with the given
model overrides and run encoder / forward / item-score loss / generate / model_step
(with backward), plus the user-memory pathway-evaluation modes when enabled.

Run BEFORE submitting new model code (an init-order bug once failed six jobs).
Usage (from GRID/):
  python scripts/check_model_build.py 'model.user_memory={neighbor_file:...,p_hist:0.2}' ...
"""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, ".")
import src.utils  # noqa: F401,E402  (same import order as src.train, avoids a cycle)
import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

BASE = [
    "experiment=tiger_train_flat",
    "data_dir=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1",
    "semantic_id_path=embeddings/beauty_rqvae/merged_predictions_tensor.pt",
    "num_hierarchies=4",
]


def main() -> int:
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=BASE + sys.argv[1:])
    model = hydra.utils.instantiate(cfg.model)
    um = getattr(model, "user_memory", None)
    mk = getattr(model, "munkey_memory", None)
    print("built", type(model).__name__, "| item_memory", model.item_memory is not None,
          "| user_memory", um is not None, "| munkey_memory", mk is not None)
    inp = model.codebooks[:6].long().reshape(1, -1).repeat(2, 1)  # 2 users x 6 items
    mask = torch.ones_like(inp)
    fut = model.codebooks[10:12].long()
    uid = None
    if um is not None or mk is not None:
        lut = (um if um is not None else mk).uid_to_row
        uid = (lut >= 0).nonzero()[:2].flatten().view(2, 1).repeat(1, 2)  # 2 known users
    model.train()
    out = model.forward(attention_mask_encoder=mask, input_ids=inp, user_id=uid, future_ids=fut)
    print("forward ok", tuple(out.shape))
    ts = {"sequence_data": inp}
    if uid is not None:
        ts["user_id"] = uid
    _, loss = model.model_step(model_input=NS(mask=mask, transformed_sequences=ts),
                               label_data=NS(labels={"sequence_data": fut.reshape(-1)}))
    loss.backward()
    print("model_step ok, loss", round(float(loss), 3))

    def unused(tag):
        # exactly what DDP rejects: a trainable parameter outside the graph
        bad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        print(f"  [{tag}] params without grad (DDP would fail):", bad[:8] if bad else "none")
    unused("default draw")
    for name, m in (("item_memory", model.item_memory), ("user_memory", um)):
        if m is not None:
            print(f"  {name} grad > 0:", bool(m.values.weight.grad.abs().sum() > 0))
    im = model.item_memory
    if im is not None and getattr(im, "recipe", "ours") == "munkey":
        for pn, pd in ((1.0, 0.0), (0.0, 0.999), (0.0, 0.0)):
            im.p_nbr, im.p_drop = pn, pd
            for p_ in model.parameters():
                p_.grad = None
            _, l3 = model.model_step(model_input=NS(mask=mask, transformed_sequences=ts),
                                     label_data=NS(labels={"sequence_data": fut.reshape(-1)}))
            l3.backward()
            unused(f"item recipe=munkey p_nbr={pn} p_drop={pd}")
    if mk is not None:
        g = mk.table.weight.grad
        print("  munkey table grad > 0:", g is not None and bool(g.abs().sum() > 0),
              "| adapter grad > 0:", bool(mk.adapter.weight.grad.abs().sum() > 0))
        # retrieval branch + both pathway drops, forced
        for kw in ({"p": 1.0}, {"tok": 0.99}, {"hist": 0.99}):
            mk.p_retrieval = kw.get("p", 0.0)
            model._munkey_p_token_drop, model._munkey_p_hist_drop = kw.get("tok", 0.0), kw.get("hist", 0.0)
            model.zero_grad()
            _, l2 = model.model_step(model_input=NS(mask=mask, transformed_sequences=ts),
                                     label_data=NS(labels={"sequence_data": fut.reshape(-1)}))
            for p_ in model.parameters():
                p_.grad = None
            l2.backward()
            print(f"  munkey train branch {kw}: loss {round(float(l2), 3)}, table grad",
                  bool(mk.table.weight.grad is not None and mk.table.weight.grad.abs().sum() > 0))
            unused(f"munkey {kw}")
        model.eval()
        with torch.no_grad():
            q = model._munkey_query_keys(inp)
            before = mk.retrieve(q, exclude_self=mk.uid_to_row[uid[:, 0]])
            top = (q @ mk.keys.T)
            top[torch.arange(2), mk.uid_to_row[uid[:, 0]]] = -float("inf")
            nn_rows = top.topk(mk.topk, dim=-1).indices.flatten()
            mk.delete(nn_rows)
            after = mk.retrieve(q, exclude_self=mk.uid_to_row[uid[:, 0]])
            print("  munkey: deleting the retrieved neighbours changes the token:",
                  bool((before - after).abs().sum() > 0), "| deleted", int(mk.deleted.sum()))
    model.eval()
    modes = (None, "no_hist", "no_token") if (um is not None or mk is not None) else (None,)
    for mode in modes:
        if um is not None:
            model._pathway_eval = mode
        if mk is not None:
            model._munkey_eval_pathway = mode
        with torch.no_grad():
            g, p = model.generate(attention_mask=mask, input_ids=inp, user_id=uid)
        print(f"generate ok (pathway={mode})", tuple(g.shape),
              "sorted:", bool((p[:, :-1] >= p[:, 1:]).all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
