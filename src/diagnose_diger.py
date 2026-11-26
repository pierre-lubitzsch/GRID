"""Why DIGER's soft-SID injection fires, or does not, on REAL batches.

`test_diger_model.py` builds its history FROM `model.codebooks`, so it cannot
catch the one failure that matters in practice: the dataloader and the model
disagreeing about which SID belongs to which item. When they disagree,
`_codes_to_item_ids` returns -1 everywhere, the soft mixture is never injected,
the tokenizer gets no gradient (DDP then reports its 17 tensors as unused) and
generation scores ids the recommender never saw, which reads as nDCG@10 = 0.0.

This runs one real training batch through the real model and reports which guard
in `_inject_soft_sid_embeddings` trips, using the training dataloader the
pipeline actually builds.

    python -m src.diagnose_diger experiment=diger_train_flat \
        data_dir=src/data/amazon_data/beauty \
        semantic_id_path=embeddings/beauty_rqvae/merged_predictions_tensor.pt \
        item_content_embeddings_path=embeddings/beauty_merged_predictions_tensor_latest.pt \
        num_hierarchies=4 trainer.accelerator=cpu trainer.devices=1
"""

from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def diagnose(cfg: DictConfig) -> None:
    with pipeline_launcher(cfg) as pm:
        model = pm.model
        model.train()
        H = int(model.num_hierarchies)
        K = int(model.num_embeddings_per_hierarchy)
        log.info(f"model={type(model).__name__} H={H} K={K} "
                 f"levels={getattr(model, 'diger_levels', '?')}")
        cb = model.codebooks
        log.info(f"model.codebooks shape={tuple(cb.shape)} dtype={cb.dtype} "
                 f"range=[{int(cb.min())},{int(cb.max())}]")

        dm = pm.datamodule
        # The datamodule reads world_size / global_rank off its trainer, which
        # Lightning normally attaches during fit(). Attach it by hand so the
        # REAL train dataloader can be built outside a fit loop.
        dm.trainer = pm.trainer
        dm.setup("fit")
        # train_dataloader() returns a retry wrapper, which may itself wrap the
        # DataLoader, so descend until a real batch appears.
        obj: Any = dm.train_dataloader()
        for _ in range(4):
            nxt = next(iter(obj))
            if isinstance(nxt, (tuple, list)) and len(nxt) >= 2 and hasattr(
                    nxt[0], "transformed_sequences"):
                obj = nxt
                break
            obj = nxt
        log.info(f"batch is a {type(obj).__name__}"
                 + (f" of {len(obj)}" if isinstance(obj, (tuple, list)) else ""))
        mi, ld = obj[0], obj[1]
        seq = list(mi.transformed_sequences.values())[0]
        log.info(f"batch input_ids shape={tuple(seq.shape)} dtype={seq.dtype} "
                 f"range=[{int(seq.min())},{int(seq.max())}]")
        log.info(f"seq_len % H = {seq.size(1) % H}  "
                 "(nonzero => guard 1 returns silently, no injection ever)")
        oob = int(((seq < 0) | (seq >= K)).sum())
        log.info(f"code values outside [0,{K}): {oob} / {seq.numel()}")

        n_items = seq.size(1) // H
        codes_items = seq[:, : n_items * H].reshape(seq.size(0), n_items, H)
        ids = model._codes_to_item_ids(codes_items)
        hit = int((ids >= 0).sum())
        log.info(f"_codes_to_item_ids: {hit}/{ids.numel()} item slots resolved "
                 f"({100.0 * hit / max(1, ids.numel()):.1f}%)")
        if hit == 0:
            blk = codes_items[0, 0].tolist()
            present = bool((cb.long().cpu() == torch.tensor(blk)).all(-1).any())
            log.info(f"  first item block {blk}; present in codebooks: {present}")
            log.info(f"  codebooks[0] = {cb[0].tolist()}")

        # Does the loss actually reach the tokenizer?
        model._diger_injected = 0
        model._diger_aux = {}
        _, loss = model.model_step(mi, ld)
        log.info(f"model_step: loss={float(loss):.5f} "
                 f"injections={model._diger_injected} "
                 f"aux={sorted(model._diger_aux)}")
        model.zero_grad(set_to_none=True)
        loss.backward()
        tok = [(n, p) for n, p in model.named_parameters() if n.startswith("tokenizer.")]
        with_grad = [n for n, p in tok if p.grad is not None and p.grad.abs().sum() > 0]
        log.info(f"tokenizer tensors: {len(tok)}, with nonzero grad: {len(with_grad)}")
        if len(with_grad) < len(tok):
            missing = [n for n, p in tok if n not in set(with_grad)]
            log.info(f"  NO GRAD ({len(missing)}): {missing[:20]}")
        log.info("VERDICT: " + (
            "tokenizer is on the gradient path; DIGER is wired correctly."
            if with_grad and model._diger_injected else
            "tokenizer is NOT on the gradient path; DIGER trains as plain TIGER "
            "and its committed ids will not match what the recommender learned."))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    diagnose(cfg)


if __name__ == "__main__":
    main()
