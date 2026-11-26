"""Decode-time re-ranking baseline: ``scripts/eval_ckpt_on_test`` plus a latent re-ranker.

Ported from the testing tree (/sc-projects/sc-proj-ikim-erase/workspace/GRID_testing,
``SemanticIDEncoderDecoder.set_reranker`` / ``rerank_candidates`` in
src/models/modules/semantic_id/tiger_generation_model.py and ``install_reranker`` in
scripts/eval_ckpt_on_test.py, experiment C8). Same math; here it is installed as an
instance-level wrapper around ``model.generate`` so the main tree's model code is untouched.

Each beam candidate c of user u is re-scored as

    score(u, c) = log p_beam(c | H_u) + alpha * cos(h_u, z_c),   h_u = mean_{i in H_u} z_i,

with z an [N, d] item table row-aligned with the SID codebook (collaborative SASRec table or
content embedding). Only the top_k_for_generation beam candidates are re-ordered, so run with a
wide beam (++model.top_k_for_generation=50) and compare with the same beam without re-ranking.

Environment (so no Hydra keys are added):
    RERANK_LATENT_PATH   item table (.pt, loaded with load_dense_embeddings); unset = no re-ranking
    RERANK_ALPHA         alpha (float)
All other arguments are the usual scripts.eval_ckpt_on_test Hydra overrides.
"""

from __future__ import annotations

import os

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts import eval_ckpt_on_test as base  # noqa: E402
from src.components.unlearning.neighborhood_sampler import load_dense_embeddings  # noqa: E402


def install_reranker(model, path: str, alpha: float) -> None:
    latent = load_dense_embeddings(path).tensor
    if latent.dim() != 2 or latent.size(0) != model.codebooks.size(0):
        raise ValueError(
            f"rerank latent must be [N={model.codebooks.size(0)}, d], row-aligned with the "
            f"SID codebook; got {tuple(latent.shape)}"
        )
    z_cpu = torch.nn.functional.normalize(latent.float(), dim=1)
    orig_generate = model.generate
    H = int(model.num_hierarchies)

    @torch.no_grad()
    def rerank(input_ids, generated_ids, marginal_probs):
        if not hasattr(model, "_adaptive_sorted_keys"):
            model._build_code_to_item_index()
        device = marginal_probs.device
        z = z_cpu.to(device)
        B, K = generated_ids.shape[:2]
        cand = model._codes_to_item_ids(generated_ids.to(device).long())  # [B, K]
        seq = input_ids.to(device).long()
        hist = model._codes_to_item_ids(seq.reshape(B, -1, H))  # [B, L]
        valid = (hist >= 0).float().unsqueeze(-1)
        h = (z[hist.clamp(min=0)] * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        h = torch.nn.functional.normalize(h, dim=-1)
        sim = (z[cand.clamp(min=0)] * h.unsqueeze(1)).sum(-1)
        sim = torch.where(cand >= 0, sim, torch.zeros_like(sim))
        score = marginal_probs.reshape(B, K).float().clamp(min=1e-30).log()
        score = score + float(alpha) * sim
        return torch.softmax(score, dim=1).to(marginal_probs.dtype)

    def generate(attention_mask, input_ids, user_id=None, **kw):
        if user_id is not None:
            kw["user_id"] = user_id
        generated_ids, marginal_probs = orig_generate(
            attention_mask=attention_mask, input_ids=input_ids, **kw)
        return generated_ids, rerank(input_ids, generated_ids, marginal_probs)

    model.generate = generate
    base.command_line_logger.info(
        f"Re-ranker INSTALLED from {path}: {tuple(latent.shape)}, alpha={alpha}, "
        f"beam={getattr(model, 'top_k_for_generation', '?')}"
    )


def evaluate(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("ckpt_path is required")
    path = os.environ.get("RERANK_LATENT_PATH", "")
    alpha = float(os.environ.get("RERANK_ALPHA", "0"))
    with base.pipeline_launcher(cfg) as pm:
        base.install_decode_filter(cfg, pm.model)
        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            install_reranker(pm.model, path, alpha)
        pm.trainer.test(model=pm.model, datamodule=pm.datamodule, ckpt_path=cfg.ckpt_path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    base.extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
