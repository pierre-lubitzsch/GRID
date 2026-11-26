"""SASRec vs TIGER under the paper's protocol (rebuttal, v2 of GRID_testing/scripts/sasrec_nau.py).

Same SASRec model and the same unlearning objective as the testing-tree script
(`/sc-projects/sc-proj-ikim-erase/workspace/GRID_testing/scripts/sasrec_nau.py`, copied, not
imported, so later edits there cannot change these results). What v2 adds:

* any of beauty / sports / toys and any data seed, on the SAME poisoned datasets as TIGER
  (src/data/amazon_data/<ds>_spam[_tgt<stratum>]_seed<N>_pct1_n1; unpopular keeps the untagged
  directory, exactly as scripts/run_fig1_js_eval.py resolves it);
* the clean (retrained) SASRec per (dataset, data seed), trained on <ds>/training with the
  data seed as training seed, like TIGER's <ds>_clean_seed<N>;
* one process per (dataset, seed, stratum) cell that loads the data once and runs the six arms
  of the rebuttal contrast (Finetune, Forget+Repair, +L_sep at lambda_s 0.01 / 0.1, NAU at
  lambda_s 0.01 / 0.1 with lambda_n 0.01 and k=8 content-embedding neighbors), unlearning
  seed = data seed (one per cell, as for TIGER);
* the test queries of TIGER's Figure 2 evaluation (src/measure_neighborhood_performance.py):
  the poisoned directory's `testing` split, one query per user (history = seq[:-1],
  label = seq[-1]); neighbor = label shares the target's FIRST RQ-KMeans SID token
  (embeddings/<ds>/merged_predictions_tensor.pt, column 0), label != target; target = label
  is the target; unrelated = everything else. Per-query NDCG@10 and log p(label) are saved
  for the neighbor group (with user ids, identical order in every arm of a cell), so paired
  bootstrap differences between arms are exact.

    python -m scripts.sasrec_nau_v2 train --dataset beauty --train_dir <dir>/training --seed 2 --out <ckpt>
    python -m scripts.sasrec_nau_v2 cell --dataset beauty --seed 2 --stratum mid \
        --clean_ckpt <ckpt> --poison_ckpt <ckpt> --out_dir logs/sasrec_nau_v2/beauty/mid_seed2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("sasrec_nau_v2")
DATA = "src/data/amazon_data"
# Item content embeddings the RQ-KMeans SIDs were built from = TIGER NAU's
# unlearning.embedding_path (checked against the canonical NAU runs' config_tree.log).
EMBEDDINGS = {"beauty": "embeddings/beauty_merged_predictions_tensor_latest.pt",
              "sports": "logs/inference/runs/embeddings/sports_10321537/pickle/merged_predictions_tensor.pt",
              "toys": "logs/inference/runs/embeddings/toys_10321536/pickle/merged_predictions_tensor.pt"}
# name: (lambda_f, lambda_s, lambda_n)
ARMS = {"finetune": (0.0, 0.0, 0.0),
        "forget_repair": (0.1, 0.0, 0.0),
        "sep_ls0p01": (0.1, 0.01, 0.0),
        "sep_ls0p1": (0.1, 0.1, 0.0),
        "nau_ls0p01": (0.1, 0.01, 0.01),
        "nau_ls0p1": (0.1, 0.1, 0.01)}


def sid_path(ds: str) -> str:
    return f"embeddings/{ds}/merged_predictions_tensor.pt"


def poison_dir(ds: str, seed: int, stratum: str) -> str:
    name = (f"{ds}_spam_seed{seed}_pct1_n1" if stratum == "unpopular"
            else f"{ds}_spam_tgt{stratum}_seed{seed}_pct1_n1")
    return os.path.join(DATA, name)


def load_codes(ds: str) -> torch.Tensor:
    """[N, 4] RQ-KMeans SIDs (stored as [4, N])."""
    t = torch.load(sid_path(ds), map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = t.get("semantic_ids", t.get("tensor"))
    return (t.t() if t.shape[0] == 4 else t).long()


# --------------------------------------------------------------------------- #
# data (verbatim from the testing-tree script)
# --------------------------------------------------------------------------- #
def read_split(directory: str) -> Dict[int, List[int]]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    shards = sorted(os.path.join(directory, f) for f in os.listdir(directory)
                    if f.endswith(".tfrecord.gz"))
    if not shards:
        raise FileNotFoundError(f"no .tfrecord.gz shards in {directory}")
    out: Dict[int, List[int]] = {}
    for rec in tf.data.TFRecordDataset(shards, compression_type="GZIP"):
        ex = tf.train.Example()
        ex.ParseFromString(rec.numpy())
        f = ex.features.feature
        out[int(f["user_id"].int64_list.value[0])] = [int(x) for x in f["sequence_data"].int64_list.value]
    return out


def prefix_rows(seqs: Sequence[List[int]]) -> List[Tuple[List[int], int]]:
    rows = []
    for s in seqs:
        for k in range(1, len(s)):
            rows.append((s[:k], s[k]))
    return rows


def pad_left(hists: Sequence[List[int]], max_len: int, pad: int) -> torch.Tensor:
    out = torch.full((len(hists), max_len), pad, dtype=torch.long)
    for i, h in enumerate(hists):
        h = h[-max_len:]
        if h:
            out[i, max_len - len(h):] = torch.tensor(h)
    return out


# --------------------------------------------------------------------------- #
# model (verbatim)
# --------------------------------------------------------------------------- #
class SASRec(nn.Module):
    def __init__(self, num_items: int, dim: int = 64, max_len: int = 50,
                 n_layers: int = 2, n_heads: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.num_items, self.max_len, self.pad = num_items, max_len, num_items
        self.item = nn.Embedding(num_items + 1, dim, padding_idx=self.pad)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.item.weight, std=dim ** -0.5)
        with torch.no_grad():
            self.item.weight[self.pad].zero_()
        self.register_buffer(
            "causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1),
            persistent=False)

    def user_repr(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.item(seq) * math.sqrt(self.item.embedding_dim)
        h = self.drop(h + self.pos.weight.unsqueeze(0))
        h = self.blocks(h, mask=self.causal)
        return self.norm(h)[:, -1]

    def log_probs(self, seq: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.user_repr(seq) @ self.item.weight[: self.pad].T, dim=-1)


def save_ckpt(model: SASRec, path: str, extra: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"num_items": model.num_items, "dim": model.item.embedding_dim,
                           "max_len": model.max_len,
                           "n_layers": len(model.blocks.layers),
                           "n_heads": model.blocks.layers[0].self_attn.num_heads,
                           "dropout": model.drop.p},
                **extra}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_ckpt(path: str, device: str) -> SASRec:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = SASRec(**obj["config"])
    model.load_state_dict(obj["state_dict"])
    return model.to(device)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
class Queries:
    """TIGER's Figure 2 test queries for one cell (fixed order: sorted user id)."""

    def __init__(self, test: Dict[int, List[int]], codes: torch.Tensor, target: int):
        self.users = sorted(u for u, s in test.items() if len(s) >= 2)
        self.hist = [test[u][:-1] for u in self.users]
        self.lab = torch.tensor([test[u][-1] for u in self.users])
        first = codes[:, 0]
        is_t = self.lab == target
        is_n = (first[self.lab] == first[target]) & ~is_t
        self.group = np.where(is_t.numpy(), "target", np.where(is_n.numpy(), "neighbor", "unrelated"))
        self.target = target

    def counts(self) -> dict:
        return {g: int((self.group == g).sum()) for g in ("neighbor", "unrelated", "target")}


@torch.no_grad()
def evaluate(model: SASRec, q: Queries, device: str, bs: int = 1024) -> dict:
    model.eval()
    nd, lp_lab, sh = [], [], []
    for i in range(0, len(q.users), bs):
        x = pad_left(q.hist[i:i + bs], model.max_len, model.pad).to(device)
        lp = model.log_probs(x)
        y = q.lab[i:i + bs].to(device)
        top = lp.topk(10, dim=1).indices
        hit = top == y[:, None]
        rank = hit.float().argmax(1)
        nd.append((hit.any(1).float() / torch.log2(rank.float() + 2)).cpu())
        lp_lab.append(lp.gather(1, y[:, None]).squeeze(1).cpu())
        sh.append((top == q.target).any(1).cpu())
    model.train()
    nd, lp_lab, sh = torch.cat(nd).numpy(), torch.cat(lp_lab).numpy(), torch.cat(sh).numpy()
    nontgt = q.group != "target"
    out = {"n": len(q.users), "ndcg@10": float(nd.mean()), "recall@10": float((nd > 0).mean()),
           # SH@10: share of queries whose label is not the target but whose top 10 holds it
           "SH@10": float(sh[nontgt].mean()), "groups": {}}
    for g in ("neighbor", "unrelated", "target"):
        m = q.group == g
        out["groups"][g] = {"n": int(m.sum()),
                            "ndcg@10": float(nd[m].mean()) if m.any() else None,
                            "logp": float(lp_lab[m].mean()) if m.any() else None,
                            "sh@10": float(sh[m].mean()) if m.any() else None}
    m = q.group == "neighbor"
    out["neighbor_per_query"] = {"user_id": [q.users[i] for i in np.flatnonzero(m)],
                                 "ndcg": nd[m].tolist(), "logp": lp_lab[m].tolist()}
    return out


# --------------------------------------------------------------------------- #
# training (as in the testing tree: early stopping on <ds>/evaluation NDCG@10)
# --------------------------------------------------------------------------- #
def train_model(ds: str, train_dir: str, seed: int, out: str, device: str, *,
                dim=64, max_len=50, n_layers=2, n_heads=1, dropout=0.2,
                epochs=200, patience=10, batch_size=256, lr=1e-3) -> None:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    num_items = load_codes(ds).shape[0]
    train = read_split(train_dir)
    val = read_split(os.path.join(DATA, ds, "evaluation"))
    rows = prefix_rows(list(train.values()))
    log.info("%s train %s: %d users, %d rows; val %d users; %d items", ds, train_dir,
             len(train), len(rows), len(val), num_items)
    vusers = [u for u, s in val.items() if len(s) >= 2]
    vX = pad_left([val[u][:-1] for u in vusers], max_len, num_items)
    vY = torch.tensor([val[u][-1] for u in vusers])
    model = SASRec(num_items, dim, max_len, n_layers, n_heads, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = pad_left([h for h, _ in rows], max_len, model.pad)
    Y = torch.tensor([y for _, y in rows])
    best, bad, best_state, best_ep = -1.0, 0, None, -1
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(rows))
        tot = 0.0
        for i in range(0, len(rows), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.nll_loss(model.log_probs(X[idx].to(device)), Y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            nd = 0.0
            for i in range(0, len(vusers), 1024):
                lp = model.log_probs(vX[i:i + 1024].to(device))
                hit = lp.topk(10, 1).indices == vY[i:i + 1024, None].to(device)
                nd += (hit.any(1).float() / torch.log2(hit.float().argmax(1).float() + 2)).sum().item()
        model.train()
        nd /= len(vusers)
        log.info("epoch %3d loss %.4f val ndcg@10 %.5f", ep, tot / len(rows), nd)
        if nd > best:
            best, bad, best_ep = nd, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    save_ckpt(model, out, {"best_epoch": best_ep, "val_ndcg@10": best, "train_dir": train_dir,
                           "seed": seed, "dataset": ds, "train_seconds": round(time.time() - t0, 1)})
    log.info("saved %s (best epoch %d, val ndcg %.5f)", out, best_ep, best)


def cmd_train(a) -> int:
    if os.path.exists(a.out):
        log.info("[skip] %s exists", a.out)
        return 0
    train_model(a.dataset, a.train_dir, a.seed, a.out, a.device)
    return 0


# --------------------------------------------------------------------------- #
# unlearning (objective verbatim from the testing tree)
# --------------------------------------------------------------------------- #
def load_item_table(path: str) -> torch.Tensor:
    o = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(o, dict):
        mat = o.get("embeddings", o.get("tensor")).float()
        ids = o.get("item_ids")
        if ids is not None:
            out = torch.empty_like(mat)
            out[torch.as_tensor(ids).long()] = mat
            mat = out
        return mat
    return o.float()


def emb_neighbors(emb: torch.Tensor, targets: Sequence[int], k: int) -> Dict[int, List[int]]:
    mat = F.normalize(emb, dim=1)
    out = {}
    for t in targets:
        s = mat @ mat[int(t)]
        for e in set(int(x) for x in targets):
            s[e] = -float("inf")
        out[int(t)] = s.topk(k).indices.tolist()
    return out


def unlearn(model: SASRec, forget: Dict[int, List[int]], retain: Dict[int, List[int]],
            targets: Sequence[int], nbrs: Dict[int, List[int]], lf: float, ls: float, ln: float,
            seed: int, dev: str, *, lambda_r=1.0, chunks=10, n_epochs=4, retain_mult=16,
            batch_size=512, lr=1e-4) -> dict:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.train()
    retain_users = [u for u, s in retain.items() if len(s) >= 2]
    fusers = sorted(forget)
    random.Random(seed).shuffle(fusers)
    parts = [fusers[i::chunks] for i in range(chunks)]
    tgt_t = torch.tensor(list(targets), device=dev)
    t0 = time.time()
    log_stats = []
    for c, cu in enumerate(parts):
        frows = prefix_rows([forget[u] for u in cu])
        n_r = retain_mult * len(frows)
        rrows = []
        rng = random.Random(seed * 1000 + c)
        while len(rrows) < n_r:
            s = retain[rng.choice(retain_users)]
            k = rng.randrange(1, len(s))
            rrows.append((s[:k], s[k]))
        fX = pad_left([h for h, _ in frows], model.max_len, model.pad)
        fY = torch.tensor([y for _, y in frows])
        rX = pad_left([h for h, _ in rrows], model.max_len, model.pad)
        rY = torch.tensor([y for _, y in rrows])
        fb = [(fX[i:i + batch_size], fY[i:i + batch_size]) for i in range(0, len(frows), batch_size)]
        rb = [(rX[i:i + batch_size], rY[i:i + batch_size]) for i in range(0, len(rrows), batch_size)]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _step in range(n_epochs):
            opt.zero_grad(set_to_none=True)
            stats = {"forget": 0.0, "nbr": 0.0, "retain": 0.0, "sep": 0.0}
            for x, y in fb:
                x, y = x.to(dev), y.to(dev)
                lp = model.log_probs(x)
                loss = torch.zeros((), device=dev)
                if lf:
                    l_f = -F.nll_loss(lp, y)
                    loss = loss + lf * l_f
                    stats["forget"] += float(l_f) / len(fb)
                if ln and nbrs:
                    rows = (y[:, None] == tgt_t[None]).any(1)
                    if rows.any():
                        idx = torch.tensor([nbrs[int(t)] for t in y[rows].tolist()], device=dev)
                        l_n = -torch.logsumexp(lp[rows].gather(1, idx), dim=1).mean()
                        loss = loss + ln * l_n
                        stats["nbr"] += float(l_n) / len(fb)
                if loss.requires_grad:
                    (loss / len(fb)).backward()
            for x, y in rb:
                x, y = x.to(dev), y.to(dev)
                lp = model.log_probs(x)
                l_r = F.nll_loss(lp, y)
                loss = lambda_r * l_r
                stats["retain"] += float(l_r) / len(rb)
                if ls:
                    keep = ~(y[:, None] == tgt_t[None]).any(1)
                    logits = torch.cat([lp[keep].gather(1, y[keep, None]), lp[keep][:, tgt_t]], dim=1)
                    l_s = (torch.logsumexp(logits, 1) - logits[:, 0]).mean()
                    loss = loss + ls * l_s
                    stats["sep"] += float(l_s) / len(rb)
                (loss / len(rb)).backward()
            opt.step()
        log_stats.append({"chunk": c, "forget_rows": len(frows), "retain_rows": len(rrows),
                          **{k: round(v, 4) for k, v in stats.items()}})
    return {"unlearn_seconds": round(time.time() - t0, 1), "chunks": log_stats,
            "hparams": {"lambda_r": lambda_r, "lambda_f": lf, "lambda_s": ls, "lambda_n": ln,
                        "chunks": chunks, "n_epochs": n_epochs, "retain_mult": retain_mult,
                        "batch_size": batch_size, "lr": lr, "seed": seed}}


def cmd_cell(a) -> int:
    dev = a.device
    os.makedirs(a.out_dir, exist_ok=True)
    pdir = poison_dir(a.dataset, a.seed, a.stratum)
    man = json.load(open(os.path.join(pdir, "forget_manifest.json")))
    targets = [int(t) for t in man["target_items"]]
    if len(targets) != 1:
        raise ValueError(f"expected one target, got {targets}")
    codes = load_codes(a.dataset)
    test = read_split(os.path.join(pdir, "testing"))
    q = Queries(test, codes, targets[0])
    meta = {"dataset": a.dataset, "seed": a.seed, "stratum": a.stratum, "target_item": targets[0],
            "poison_dir": pdir, "sid_path": sid_path(a.dataset), "counts": q.counts(),
            "definition": "neighbor: test next item shares the target's first RQ-KMeans SID token; "
                          "target: next item IS the target; unrelated: the rest (TIGER Figure 2 queries, "
                          "full population, no 256 cap)",
            "clean_ckpt": os.path.realpath(a.clean_ckpt), "poison_ckpt": os.path.realpath(a.poison_ckpt)}
    log.info("cell %s %s seed %d: target %d, %s", a.dataset, a.stratum, a.seed, targets[0], q.counts())
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    for name, ck in (("clean", a.clean_ckpt), ("poison", a.poison_ckpt)):
        out = os.path.join(a.out_dir, f"{name}.json")
        if os.path.exists(out):
            continue
        r = evaluate(load_ckpt(ck, dev), q, dev)
        r["ckpt"] = os.path.realpath(ck)
        json.dump(r, open(out, "w"))
    forget = read_split(os.path.join(pdir, "training_forget"))
    retain = read_split(os.path.join(pdir, "training_retain"))
    emb = load_item_table(EMBEDDINGS[a.dataset])
    if emb.shape[0] != codes.shape[0]:
        raise ValueError(f"embedding rows {emb.shape[0]} != items {codes.shape[0]}")
    nbrs = emb_neighbors(emb, targets, a.nbr_k)
    log.info("forget users %d, retain users %d, neighbors %s", len(forget), len(retain), nbrs)
    for arm in a.arms.split(","):
        out = os.path.join(a.out_dir, f"{arm}.json")
        if os.path.exists(out):
            log.info("[skip] %s", out)
            continue
        lf, ls, ln = ARMS[arm]
        model = load_ckpt(a.poison_ckpt, dev)
        info = unlearn(model, forget, retain, targets, nbrs if ln else {}, lf, ls, ln, a.seed, dev,
                       n_epochs=a.n_epochs)
        r = evaluate(model, q, dev)
        r.update(info)
        r.update({"arm": arm, "nbr_k": a.nbr_k, "nbrs": {str(k): v for k, v in nbrs.items()} if ln else None,
                  "poison_ckpt": os.path.realpath(a.poison_ckpt)})
        json.dump(r, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        log.info("%s: ndcg %.5f SH %.5f nbr ndcg %s logp %s (%.1fs)", arm, r["ndcg@10"], r["SH@10"],
                 r["groups"]["neighbor"]["ndcg@10"], r["groups"]["neighbor"]["logp"], r["unlearn_seconds"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True, choices=tuple(EMBEDDINGS))
    t.add_argument("--train_dir", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default=dev)
    c = sub.add_parser("cell")
    c.add_argument("--dataset", required=True, choices=tuple(EMBEDDINGS))
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--stratum", required=True, choices=("unpopular", "mid", "popular"))
    c.add_argument("--clean_ckpt", required=True)
    c.add_argument("--poison_ckpt", required=True)
    c.add_argument("--out_dir", required=True)
    c.add_argument("--arms", default=",".join(ARMS))
    c.add_argument("--nbr_k", type=int, default=8)
    c.add_argument("--n_epochs", type=int, default=4)
    c.add_argument("--device", default=dev)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return {"train": cmd_train, "cell": cmd_cell}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
