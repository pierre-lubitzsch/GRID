"""Mean-output ablation of PKM memories: how much TOKEN-SPECIFIC information does a
(collapsed or balanced) memory contribute?

For every HashingMemory in the model:
  1. average its output over the real (non-padding) encoder tokens of validation rows;
  2. evaluate the test split with the output replaced by that mean (keeps the constant
     shift the network was trained to expect, removes only the input dependence), by
     zero (also removes the shift), and unablated.
A small mean-ablation drop means the memory carries little token-specific signal.
Encoder memories only (the mask comes from the encoder call).

Usage (from GRID/; CPU is fine):
  python -m scripts.pkm_mean_ablation --ckpt <last.ckpt> \
      --overrides "model.pkm_layers={encoder:[1],decoder:null} model.pkm_mode=replace" \
      --data src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1 --out <json>
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
import src.utils  # noqa: F401,E402  (import order as in src.train)
from src.utils.custom_hydra_resolvers import *  # noqa: F401,F403,E402  (config resolvers)
import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--overrides", default="", help="space-separated model overrides")
    ap.add_argument("--data", required=True, help="dataset dir (testing/, evaluation/, manifest)")
    ap.add_argument("--sid", default="embeddings/beauty_rqvae/merged_predictions_tensor.pt")
    ap.add_argument("--val_rows", type=int, default=2000)
    ap.add_argument("--test_rows", type=int, default=6000)
    ap.add_argument("--modes", default="none,mean,zero")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.components.unlearning.val_selection import build_val_batches, evaluate_batches
    from src.models.components.network_blocks.product_key_memory import (
        HashingMemory, resolve_auto_query_norm)

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            "experiment=tiger_train_flat", f"data_dir={a.data}", f"semantic_id_path={a.sid}",
            "num_hierarchies=4", *a.overrides.split()])
    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    resolve_auto_query_norm(model, sd, adopt=True)
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        print("WARNING load: missing", r.missing_keys[:5], "unexpected", r.unexpected_keys[:5])
    mems = {n: m for n, m in model.named_modules() if isinstance(m, HashingMemory)}
    if not mems:
        raise SystemExit("no HashingMemory in this model")
    dl = hydra.utils.instantiate(cfg.data_loading.train_dataloader_config.dataloader)
    dev = torch.device("cpu")
    val = build_val_batches(dl, os.path.join(a.data, "evaluation"), dev, max_rows=a.val_rows)
    test = build_val_batches(dl, os.path.join(a.data, "testing"), dev, max_rows=a.test_rows)

    st = {"mode": None, "mask": None, "sum": {}, "n": {}, "mean": {}}

    def enc_pre(_m, args, kwargs):
        st["mask"] = kwargs.get("attention_mask")

    def mem_hook(name):
        def hook(_m, _inp, out):
            if st["mode"] is None or not torch.is_tensor(out) or out.dim() != 3:
                return None
            mask = st["mask"]
            valid = (mask.bool() if mask is not None and tuple(mask.shape) == tuple(out.shape[:2])
                     else torch.ones(out.shape[:2], dtype=torch.bool))
            if st["mode"] == "collect":
                st["sum"][name] = st["sum"].get(name, 0) + out[valid].detach().double().sum(0)
                st["n"][name] = st["n"].get(name, 0) + int(valid.sum())
                return None
            if st["mode"] == "mean":
                return st["mean"][name].to(out.dtype).expand_as(out).clone()
            if st["mode"] == "zero":
                return torch.zeros_like(out)
            return None
        return hook

    model.encoder.register_forward_pre_hook(enc_pre, with_kwargs=True)
    for n, m in mems.items():
        m.register_forward_hook(mem_hook(n))

    t0 = time.time()
    st["mode"] = "collect"
    evaluate_batches(model, val)
    for n in mems:
        if not st["n"].get(n):
            raise SystemExit(f"memory {n} produced no 3-D encoder output to average")
        st["mean"][n] = (st["sum"][n] / st["n"][n]).float()
    res = {"ckpt": a.ckpt, "memories": list(mems), "val_tokens": st["n"],
           "test_rows": sum(int(b[0].mask.shape[0]) for b in test), "results": {}}
    for mode in a.modes.split(","):
        st["mode"] = None if mode == "none" else mode
        mtr = evaluate_batches(model, test)
        res["results"][mode] = {k: mtr[k] for k in ("ndcg@10", "SH@10", "recall@10") if k in mtr}
        print(f"[ablation] {mode:5s} {res['results'][mode]}  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
