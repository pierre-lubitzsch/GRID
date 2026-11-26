"""TIGER's semantic-ID tokenizer (Rajput et al. 2023, Sec. 4.1 / App. A), standalone.

RQ-VAE on Sentence-T5 item embeddings (768-d):
  encoder 768 -> 512 -> 256 -> 128 -> 32 (ReLU), decoder mirrored;
  3 residual levels, codebook 256 x 32 each, k-means initialization on the first batch;
  loss = ||x - x_hat||^2 + sum_d ||sg[r_d] - e_d||^2 + beta ||r_d - sg[e_d]||^2 (squared norms,
  summed over dimensions, batch mean), beta 0.25,
  straight-through estimator; Adagrad lr 0.4 (TF's initial accumulator 0.1), batch 1024, 20k epochs.
Semantic ID = the 3 codes + a 4th code that makes colliding items unique (0, 1, 2, ... in item-id
order within a collision group; items without a collision get 0).

Output: <out_dir>/merged_predictions_tensor.pt, int64 (4, num_items), the layout of
embeddings/<ds>_rqvae/merged_predictions_tensor.pt; plus rqvae.pt (weights) and stats.json.

Usage (from GRID/): python -m scripts.tiger_rqvae --emb embeddings/beauty_st5_tensor.pt \
    --out_dir embeddings/beauty_tiger_rqvae
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def sqnorm(d: torch.Tensor) -> torch.Tensor:
    """Squared L2 norm per row, averaged over the batch."""
    return d.pow(2).sum(-1).mean()


def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(x.size(0), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


class RQVAE(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 256, 128), latent=32, levels=3, codes=256, beta=0.25):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim])
        self.codebooks = nn.ParameterList([nn.Parameter(torch.zeros(codes, latent)) for _ in range(levels)])
        self.beta = beta
        self.initialized = False

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor, seed: int):
        r = self.encoder(x)
        for d, cb in enumerate(self.codebooks):
            cb.data.copy_(kmeans(r, cb.size(0), seed=seed + d))
            r = r - cb[torch.cdist(r, cb).argmin(1)]
        self.initialized = True

    def quantize(self, z):
        r, zq, codes, qloss = z, torch.zeros_like(z), [], z.new_zeros(())
        for cb in self.codebooks:
            idx = torch.cdist(r, cb).argmin(1)
            e = cb[idx]
            # squared norms summed over dimensions, as the paper writes them (||.||^2); a
            # per-element mean weakens the reconstruction term 768/32 = 24x relative to these
            # terms, and the first level then collapses to a single code
            qloss = qloss + sqnorm(e - r.detach()) + self.beta * sqnorm(r - e.detach())
            zq = zq + e
            r = r - e.detach()
            codes.append(idx)
        return zq, torch.stack(codes, 1), qloss

    def forward(self, x):
        z = self.encoder(x)
        zq, codes, qloss = self.quantize(z)
        xr = self.decoder(z + (zq - z).detach())  # straight-through
        rec = sqnorm(xr - x)
        return rec, qloss, codes


def add_collision_code(codes3: torch.Tensor):
    """(N, 3) -> (N, 4): 4th code = rank within the group of items sharing all 3 codes."""
    n = codes3.size(0)
    fourth = torch.zeros(n, dtype=torch.long)
    seen = {}
    for i, key in enumerate(map(tuple, codes3.tolist())):
        fourth[i] = seen.get(key, 0)
        seen[key] = fourth[i].item() + 1
    return torch.cat([codes3, fourth.unsqueeze(1)], 1), max(seen.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adagrad_init_acc", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=500)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(torch.load(a.emb, map_location="cpu", weights_only=False)).float().to(dev)
    n = x.size(0)
    model = RQVAE(in_dim=x.size(1)).to(dev)
    # TIGER was built on TensorFlow, whose Adagrad starts the accumulator at 0.1; PyTorch's
    # default 0 makes the first step +-lr = +-0.4 per weight, which diverges at once
    opt = torch.optim.Adagrad(model.parameters(), lr=a.lr, initial_accumulator_value=a.adagrad_init_acc)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    t0 = time.time()
    print(f"[rqvae] {n} items x {x.size(1)}, {a.epochs} epochs, batch {a.batch}, Adagrad lr {a.lr}, {dev}",
          flush=True)
    for ep in range(a.epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.batch):
            xb = x[perm[i:i + a.batch]]
            if not model.initialized:
                model.init_codebooks(xb, a.seed)
            rec, qloss, _ = model(xb)
            opt.zero_grad(set_to_none=True)
            (rec + qloss).backward()
            opt.step()
        if ep % a.log_every == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                rec, qloss, codes = model(x)
                used = [int(codes[:, d].unique().numel()) for d in range(codes.size(1))]
                uniq = int(torch.unique(codes, dim=0).size(0))
            print(f"[rqvae] ep {ep:5d} rec {rec.item():.5f} q {qloss.item():.5f} codes used {used} "
                  f"unique 3-code ids {uniq}/{n} ({time.time() - t0:.0f}s)", flush=True)

    with torch.no_grad():
        rec, qloss, codes3 = model(x)
    sids, max_group = add_collision_code(codes3.cpu())
    if max_group > 256:
        raise SystemExit(f"collision group of {max_group} items does not fit a 256-way 4th code")
    out = sids.T.contiguous()  # (4, N)
    assert torch.unique(sids, dim=0).size(0) == n, "semantic IDs are not unique"
    os.makedirs(a.out_dir, exist_ok=True)
    torch.save(out, os.path.join(a.out_dir, "merged_predictions_tensor.pt"))
    torch.save(model.state_dict(), os.path.join(a.out_dir, "rqvae.pt"))
    stats = {
        "emb": a.emb, "items": n, "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "adagrad_init_acc": a.adagrad_init_acc, "seed": a.seed,
        "final_rec_loss": rec.item(), "final_q_loss": qloss.item(),
        "codes_used_per_level": [int(codes3[:, d].unique().numel()) for d in range(3)],
        "unique_3code_ids": int(torch.unique(codes3, dim=0).size(0)),
        "items_with_collision": int((sids[:, 3] > 0).sum()) + 0,
        "max_collision_group": int(max_group),
        "max_code_per_position": out.max(1).values.tolist(),
        "wall_s": round(time.time() - t0, 1),
    }
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=2)
    print("[rqvae]", json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
