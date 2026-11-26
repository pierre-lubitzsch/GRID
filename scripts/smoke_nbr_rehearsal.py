"""CPU smoke test of the neighborhood-rehearsal opt-in (unlearning.nbr_rehearsal_rate).

No model is loaded. Checks, on Beauty mid seed 2 (request-sized retain subset of 368 sessions):
  * rows per retain batch unchanged, label_location rows renumbered, mask token at the label slot,
  * replaced share ~= rate and replaced rows carry a neighbor label,
  * the neighbor set equals the one NAU's L_n builds (_build_coherence_neighbors, embedding, k=8).
Exit code != 0 on any failure (used as an afterok gate for the GPU runs).

    python -u scripts/smoke_nbr_rehearsal.py <workdir>
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import hydra  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403

tf.config.set_visible_devices([], "GPU")
from src.components.unlearning.neighborhood_sampler import (  # noqa: E402
    _ShardWriter, collect_items_in_shards)
from src.components.unlearning.unified import label_sids  # noqa: E402
from src.models.modules.semantic_id import tiger_unlearning_module as M  # noqa: E402

ROOT = os.getcwd()
WD = os.path.abspath(sys.argv[1])
BASE = "src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1"
SID = "embeddings/beauty/merged_predictions_tensor.pt"
EMB = "embeddings/beauty_merged_predictions_tensor_latest.pt"
RATE = 0.5
target = json.load(open(os.path.join(BASE, "forget_manifest.json")))["target_items"]

with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base="1.3"):
    cfg = compose("unlearn.yaml", overrides=[
        "experiment=tiger_unlearn_unified_sequential", f"data_dir={BASE}",
        f"semantic_id_path={SID}", "num_hierarchies=4",
        f"unlearning.nbr_rehearsal_rate={RATE}", f"unlearning.embedding_path={EMB}"])
dl = hydra.utils.instantiate(cfg.data_loading.train_dataloader_config.dataloader)
ucfg = dict(cfg.unlearning)

req = os.path.join(WD, "req")
shutil.rmtree(req, ignore_errors=True)
os.makedirs(req)
os.symlink(os.path.abspath(os.path.join(BASE, "training_retain")), os.path.join(req, "training_retain"))
sub = os.path.join(WD, "sub")
shutil.rmtree(sub, ignore_errors=True)
w = _ShardWriter(sub, "data", 4096)
for i, r in enumerate(tf.data.TFRecordDataset(
        M._list_shards_safe(os.path.join(BASE, "training_retain")), compression_type="GZIP")):
    if i >= 368:
        break
    w.write(r.numpy())
w.close()
bs = ucfg.get("batch_size_per_device")
retain = M._drain_loader(M._build_finite_loader(base_train_cfg=dl, data_folder=sub,
                                                batch_size_per_device_override=bs), torch.device("cpu"))
fdir = os.path.join(BASE, "training_forget")
forget = M._drain_loader(M._build_finite_loader(base_train_cfg=dl, data_folder=fdir,
                                                batch_size_per_device_override=bs), torch.device("cpu"))
vis = collect_items_in_shards(M._list_shards_safe(fdir))
orig = [label_sids(b, 4).clone() for b in retain]
ctx = {"retain_batches": retain, "forget_batches": forget, "visible_forget_items": vis,
       "meta": {"target_items": list(target)}}
kw = dict(semantic_id_path=SID, num_hierarchies=4, data_dir=req, retain_subdir="training_retain",
          train_dataloader_config=dl, seed=2)
M.TigerUnlearningModule._apply_nbr_rehearsal(None, ctx, ucfg, RATE, **kw)
st = ctx["meta"]["nbr_rehearsal"]
print(json.dumps(st))
cb = M.load_codebook(SID, num_hierarchies=4)
nbr = {tuple(cb[n].tolist()) for n in st["neighbor_items"]}
assert len(st["neighbor_items"]) == 8 * len(target), st["neighbor_items"]
tot_n = tot_nbr = 0
for l0, b in zip(orig, ctx["retain_batches"]):
    mi, ld = b
    n = mi.mask.shape[0]
    assert n == l0.shape[0], (n, l0.shape)
    loc = ld.label_location["sequence_data"].view(n, 4, 2)
    assert torch.equal(loc[:, :, 0], torch.arange(n).view(n, 1).expand(n, 4)), "label_location rows"
    seqs = mi.transformed_sequences["sequence_data"]
    assert (seqs[torch.arange(n), loc[:, 0, 1]] == -1).all(), "mask token not at label slot"
    assert ld.labels["sequence_data"].numel() == 4 * n
    k = sum(tuple(x) in nbr for x in label_sids(b, 4).tolist())
    k0 = sum(tuple(x) in nbr for x in l0.tolist())
    print(f"batch rows {n}: neighbor-labeled {k} (before {k0})")
    tot_n += n
    tot_nbr += k
assert tot_nbr >= int(0.95 * RATE * tot_n), (tot_nbr, tot_n)
coh = M.TigerUnlearningModule._build_coherence_neighbors(
    None, forget_batches=forget, semantic_id_path=SID, num_hierarchies=4, neighborhood_count=8,
    neighborhood_prefix_length=2, exclude_items=vis, coherence_rows="target_only",
    target_items=set(target), neighbor_method="embedding", embedding_path=EMB)
s = set()
for c in coh:
    if c is None:
        continue
    sids, mask = c
    for r in range(sids.shape[0]):
        for j in range(sids.shape[1]):
            if mask[r, j] > 0:
                s.add(tuple(sids[r, j].tolist()))
print("NAU L_n neighbor SIDs == rehearsal SIDs:", s == nbr, len(s), len(nbr))
assert s == nbr
print("SMOKE OK")
