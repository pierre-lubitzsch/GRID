"""Conformance tests for the DIGER tokenizer against the paper (arXiv 2601.19711v3).

Three properties, each of which was ambiguous or wrong before 2026-08-25, and each
of which the authors' reference repo (github.com/junchen-fu/DIGER) implements
DIFFERENTLY from the paper. The point of pinning them here is that "make it look
like the reference repo" is the obvious wrong move for all three.

TEST 1  SOFT UPDATE (paper Sec 4.1 / Fig 3). Gradients must reach EVERY codebook
        entry, weighted by its Gumbel-Softmax probability -- `e-bar = sum_i y_i e_i`.
        The paper explicitly contrasts this with STE, which sends gradient only to
        the argmax row and which the reference repo does
        (`Ind = hard - soft.detach() + soft; x_q = Ind @ W`). This is DIGER's
        defining choice, so a regression here silently turns DIGER into TIGER-with-
        extra-steps.

TEST 2  SIGMA IS THE NOISE STANDARD DEVIATION (paper Sec 4.2, Eq. 9). SDUD anneals
        `sigma* = max(0, sqrt(L_gen) - lambda)` and sigma SCALES THE GUMBEL NOISE.
        Until 2026-08-25 sigma was computed for the `L_sigma` term and discarded,
        so `sdud` mode reweighted the loss and never touched the noise -- the
        exploration schedule simply did not happen.

TEST 3  FORWARD IS THE HARD CODEWORD, exactly. The straight-through arrangement
        must be `e_hard.detach() + (e_soft - e_soft.detach())`, whose forward value
        is bit-identical to `e_hard`. The other arrangement leaks quantization error
        into the residual recursion, compounding across levels.

Run:  python test_diger_paper_conformance.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.modules.semantic_id.diger_tokenizer import (  # noqa: E402
    DigerTokenizer,
    sdud_loss,
)

C, L, K, D, B = 32, 3, 8, 16, 64
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


def mk(**kw) -> DigerTokenizer:
    torch.manual_seed(0)
    kw.setdefault("codebook_size", K)
    tok = DigerTokenizer(
        content_dim=C, n_levels=L, latent_dim=D,
        encoder_hidden=(64, 32), **kw,
    )
    torch.manual_seed(1)
    tok.fit_input_norm(torch.randn(256, C))
    return tok


def content(n: int = B) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(n, C)


# --------------------------------------------------------------------------- #
print("\nTEST 1: Soft Update — gradient reaches ALL codebook entries (Sec 4.1)")
tok = mk(use_gumbel_noise=False)
tok.train()
out = tok(content())
out.quantized.sum().backward()
g = tok.codebooks.grad
check(g is not None, "codebooks receive gradient at all")
# The decisive check: far more than one row per level must be non-zero. STE-on-the
# one-hot would touch at most the |{selected}| rows.
for lvl in range(L):
    nz = int((g[lvl].abs().sum(dim=1) > 0).sum())
    sel = len(set(out.codes[:, lvl].tolist()))
    check(nz > sel,
          f"level {lvl}: {nz}/{K} rows updated vs only {sel} selected "
          f"(Soft Update, not Hard)")
check(int((g.abs().sum(dim=(1, 2)) > 0).sum()) == L, "every level updated")

# Gradient magnitude must track the soft probability, which is what "weighted by
# their Gumbel-Softmax probabilities" means. Use a level's mean probs vs row grad.
tok2 = mk(use_gumbel_noise=False)
tok2.train()
o2 = tok2(content())
o2.quantized.sum().backward()
lvl = 0
probs = o2.probs[:, lvl].mean(dim=0)                       # [K]
rowg = tok2.codebooks.grad[lvl].abs().sum(dim=1)          # [K]
mask = probs > 0
corr = torch.corrcoef(torch.stack([probs[mask], rowg[mask]]))[0, 1]
# NOT expected to be ~1.0: `e_st` also feeds `residual = residual - e_st`, so a
# row's total gradient mixes its own level's soft weight with contributions routed
# through every LATER level's residual. A strong positive rank relationship is the
# right assertion; exact proportionality would only hold for a single-level
# quantizer. Measured 0.815 at L=3.
check(bool(corr > 0.6), f"row-grad magnitude tracks soft prob (corr={corr:.3f})")

# --------------------------------------------------------------------------- #
print("\nTEST 2: sigma is the Gumbel-noise std, and it anneals (Sec 4.2, Eq. 9)")
# Closed form, straight from the paper.
for lg, lam, want in ((4.0, 1.4, 0.6), (1.0, 1.4, 0.0), (100.0, 1.4, 8.6)):
    got = float(torch.clamp(torch.tensor(lg).sqrt() - lam, min=0.0))
    check(abs(got - want) < 1e-4, f"sigma*(L={lg}, lam={lam}) = {got:.4f}")

tok = mk(sigma_scaled_noise=True, sigma_ema_beta=1.0)
tok.train()
check(float(tok.noise_sigma) == 1.0, "sigma starts at 1.0 (unscaled)")
tok.update_noise_sigma(torch.tensor(100.0), 1.4)
big = float(tok.noise_sigma)
check(abs(big - 8.6) < 1e-3, f"high L_gen -> large sigma ({big:.3f})")
tok.update_noise_sigma(torch.tensor(1.0), 1.4)
check(float(tok.noise_sigma) == 0.0, "low L_gen -> sigma anneals to 0 (exploitation)")

# The wiring: sigma must actually change the noise magnitude.
tok = mk(sigma_scaled_noise=True, sigma_ema_beta=1.0, frq_decay_ratio=0.0)
tok.train()
logits = torch.zeros(256, K)
tok.update_noise_sigma(torch.tensor(100.0), 1.4)     # sigma = 8.6
torch.manual_seed(7); loud = tok._exploration_noise(logits, 0).std()
tok.update_noise_sigma(torch.tensor(1.0), 1.4)       # sigma = 0
torch.manual_seed(7); quiet = tok._exploration_noise(logits, 0).std()
check(float(quiet) == 0.0, f"sigma=0 -> noise identically zero (std={float(quiet)})")
check(float(loud) > 1.0, f"sigma=8.6 -> loud noise (std={float(loud):.2f})")

# sigma is a BUFFER, so it survives a checkpoint round-trip; a plain float would
# silently restart the anneal on resume.
# Clone: state_dict() hands back REFERENCES to the live buffers, and
# update_noise_sigma mutates in place, so an uncloned capture tracks the mutation
# and the test would pass vacuously. (Real checkpointing serialises, so this
# aliasing is a test-only hazard.)
sd = {k: v.clone() for k, v in tok.state_dict().items()}
check("noise_sigma" in sd, "noise_sigma is in state_dict (survives resume)")
tok.update_noise_sigma(torch.tensor(100.0), 1.4)
tok.load_state_dict(sd)
check(float(tok.noise_sigma) == 0.0, "sigma restored from checkpoint")

# Off by default => the frqud-only default path is untouched.
tok_off = mk()
check(tok_off.sigma_scaled_noise is False, "sigma scaling OFF by default")
tok_off.train()
tok_off.noise_sigma.fill_(99.0)          # would explode the noise IF it were used
torch.manual_seed(7); a = tok_off._exploration_noise(torch.zeros(256, K), 0)
tok_ref = mk(); tok_ref.train()
torch.manual_seed(7); b = tok_ref._exploration_noise(torch.zeros(256, K), 0)
check(torch.equal(a, b), "default mode ignores sigma entirely (bit-identical)")

# --------------------------------------------------------------------------- #
print("\nTEST 2b: the two THRESHOLDS the paper defines")
# (i) FrqUD hot-code threshold, paper Eq. 11-12:  gamma = r * f_bar = r / K.
#     A code receives Gumbel noise only while its usage frequency exceeds gamma;
#     low-frequency codes stay deterministic. `frq_decay_ratio` IS the paper's r.
# Vary BOTH r and K so the identity is actually exercised, not just r/8.
for r, k in ((1.5, 8), (2.0, 16), (3.0, 64)):
    tok = mk(frq_decay_ratio=r, codebook_size=k)
    got = tok.frq_decay_ratio / tok.codebook_size
    check(abs(got - r / k) < 1e-12 and tok.codebook_size == k,
          f"gamma = r/K = {r}/{k} = {got:.6f}")
tok = mk(frq_decay_ratio=2.0, use_gumbel_noise=True)
tok.train()
lvl = 0
gamma = tok.frq_decay_ratio / tok.codebook_size
# One code just above gamma, the rest well below: only that column may be noisy.
tok.code_freq[lvl].fill_(gamma * 0.5)
tok.code_freq[lvl][3] = gamma * 2.0
n = tok._exploration_noise(torch.zeros(512, K), lvl)
nz_cols = (n.abs().sum(dim=0) > 0).nonzero().flatten().tolist()
check(nz_cols == [3], f"only the above-gamma code is perturbed (cols={nz_cols})")
tok.code_freq[lvl].fill_(gamma * 0.5)            # nothing hot
check(float(tok._exploration_noise(torch.zeros(512, K), lvl).abs().sum()) == 0.0,
      "no code above gamma -> no noise at all (fully deterministic)")

# (ii) SDUD's implicit loss threshold: sigma* = max(0, sqrt(L)-lambda) is ZERO
#      for every L_gen <= lambda^2, so exploration switches off below that.
lam = 1.4
thr = lam ** 2
for lg, expect_noise in ((1.0, False), (thr - 1e-3, False), (thr, False),
                         (thr + 1e-2, True), (4.0, True)):
    sig = float(torch.clamp(torch.tensor(float(lg)).sqrt() - lam, min=0.0))
    check((sig > 0) == expect_noise,
          f"L_gen={lg:.4f} vs lambda^2={thr:.4f} -> sigma*={sig:.6f} "
          f"({'explore' if sig > 0 else 'exploit'})")

# --------------------------------------------------------------------------- #
print("\nTEST 3: forward value is EXACTLY the hard codeword")
tok = mk(use_gumbel_noise=False, normalize_residuals=False)
tok.eval()
with torch.no_grad():
    o = tok(content())
    # Rebuild the hard quantization independently and require bit-equality.
    cn = tok._normalize_input(content())
    z = tok.encoder(cn)
    resid, manual = z, torch.zeros_like(z)
    for lvl in range(L):
        lg = tok.assignment_logits(resid, lvl)
        h = lg.argmax(dim=-1)
        e = tok.codebooks[lvl][h]
        manual = manual + e
        resid = resid - e
check(torch.equal(o.quantized, manual),
      "quantized == sum of hard codewords, bit-for-bit")

# --------------------------------------------------------------------------- #
print("\nTEST 4: balance loss is opt-in and inert at 0 (NOT in the paper)")
t0 = mk(use_gumbel_noise=False)
t0.train()
v0 = float(t0(content()).vq_loss)
t1 = mk(use_gumbel_noise=False, balance_weight=0.0)
t1.train()
check(abs(float(t1(content()).vq_loss) - v0) < 1e-9, "balance_weight=0 changes nothing")
t2 = mk(use_gumbel_noise=False, balance_weight=1.0)
t2.train()
v2 = float(t2(content()).vq_loss)
check(v2 > v0, f"balance_weight=1 adds a positive penalty ({v2:.4f} > {v0:.4f})")

# --------------------------------------------------------------------------- #
print("\nTEST 5: sdud_loss still matches paper Eq. 9 at the closed-form optimum")
lg = torch.tensor(4.0)
lam = 1.4
val = float(sdud_loss(lg, lam=lam))
sig = max(0.0, float(lg.sqrt()) - lam)
want = float(lg) / (2 * (sig + lam) ** 2) + torch.log(torch.tensor(sig + lam))
check(abs(val - float(want)) < 1e-6, f"L_sigma = {val:.5f} matches the formula")

print("\n" + "=" * 70)
if FAILURES:
    print(f"FAILED: {len(FAILURES)}")
    for f in FAILURES:
        print("  -", f)
    raise SystemExit(1)
print("All DIGER paper-conformance checks passed.")
