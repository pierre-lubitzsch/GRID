"""Tests for validation-based epoch selection (src/components/unlearning/val_selection.py).

The validation metrics are scripted per epoch (evaluate_batches is replaced), so
the ranking rule, patience, and best-epoch restore are checked without a
checkpoint or data, both on the selector alone and through unified_unlearn's
epoch_end_callback with the mock model from test_unified_logic.

    python test_val_selection.py
"""

from __future__ import annotations

import sys

import torch

import src.components.unlearning.val_selection as vs
from test_unified_logic import MockTigerModel, _make_batches  # also patches batch helpers
from src.components.unlearning.unified import unified_unlearn

REF_SH, REF_NDCG = 0.0, 0.05


def _scripted(model, script):
    """Make evaluate_batches return script[epoch-1]; record the bias at each call."""
    calls = []

    def fake(_model, _batches):
        sh, ndcg = script[len(calls)]
        calls.append(float(model.bias_.detach()))
        return {"SH@10": sh, "ndcg@10": ndcg, "loss": 0.0}

    vs.evaluate_batches = fake
    return calls


def _selector(model, **kw):
    return vs.EpochSelector(model=model, batches=[], ref_sh=REF_SH, ref_ndcg=REF_NDCG, **kw)


def test_sh_tie_prefers_higher_tau_and_patience_stops():
    m = MockTigerModel()
    script = [(0.010, 0.049), (0.0, 0.048), (0.0, 0.049), (0.0, 0.0485), (0.0, 0.047)]
    calls = _scripted(m, script)
    sel = _selector(m, patience=2, max_epochs=10)
    stops = []
    for e in range(1, 6):
        with torch.no_grad():
            m.bias_.fill_(float(e))      # weights identify the epoch
        stops.append(sel(e))
    assert stops == [False, False, False, False, True], stops
    assert sel.best_epoch == 3, sel.best_epoch          # SH tie with 2, higher tau_P
    rec = sel.finalize()
    assert float(m.bias_) == 3.0, float(m.bias_)       # best weights restored
    assert rec["epochs_run"] == 5 and len(calls) == 5


def test_feasible_beats_lower_sh():
    m = MockTigerModel()
    _scripted(m, [(0.0, 0.045), (0.005, 0.0495)])      # tau 0.90 vs 0.99
    sel = _selector(m, tau_floor=0.95)
    sel(1); sel(2)
    assert sel.best_epoch == 2, sel.best_epoch


def test_all_infeasible_falls_back_to_lowest_sh():
    m = MockTigerModel()
    _scripted(m, [(0.010, 0.040), (0.002, 0.030), (0.004, 0.045)])
    sel = _selector(m, tau_floor=0.95, patience=5)
    for e in (1, 2, 3):
        sel(e)
    assert sel.best_epoch == 2, sel.best_epoch


def test_sh_tie_tolerance():
    m = MockTigerModel()
    _scripted(m, [(0.0010, 0.048), (0.0012, 0.0495)])
    sel = _selector(m, sh_tie_tol=0.0005)
    sel(1); sel(2)
    assert sel.best_epoch == 2                          # within tol -> higher tau_P
    m2 = MockTigerModel()
    _scripted(m2, [(0.0010, 0.048), (0.0012, 0.0495)])
    sel2 = _selector(m2, sh_tie_tol=0.0)
    sel2(1); sel2(2)
    assert sel2.best_epoch == 1                         # exact: lower SH wins


def test_max_epochs_and_min_epochs():
    m = MockTigerModel()
    _scripted(m, [(0.01 - 0.001 * e, 0.049) for e in range(1, 4)])
    sel = _selector(m, max_epochs=3, patience=99)
    assert [sel(e) for e in (1, 2, 3)] == [False, False, True]
    m2 = MockTigerModel()
    _scripted(m2, [(0.0, 0.049), (0.005, 0.049)])
    sel2 = _selector(m2, min_epochs=2)
    sel2(1); sel2(2)
    assert sel2.best_epoch == 2                         # epoch 1 not a candidate


def test_through_unified_loop():
    torch.manual_seed(0)
    m = MockTigerModel()
    # optim_steps_per_pass = min(3, 6) = 3, so the callback fires every 3 steps.
    script = [(0.010, 0.049), (0.002, 0.049), (0.004, 0.049), (0.006, 0.049), (0.0, 0.049)]
    calls = _scripted(m, script)
    sel = _selector(m, patience=2, max_epochs=8)
    info = unified_unlearn(m, _make_batches(3), _make_batches(6), n_epochs=8, lr=1e-2,
                           epoch_end_callback=sel, device=torch.device("cpu"))
    assert info["steps"] == 24 and info["steps_run"] == 12, (info["steps"], info["steps_run"])
    assert len(calls) == 4 and sel.best_epoch == 2
    after_loop = float(m.bias_)
    sel.finalize()
    assert float(m.bias_) == calls[1] != after_loop     # epoch-2 weights restored


def test_unified_without_callback_unchanged():
    info = unified_unlearn(MockTigerModel(), _make_batches(3), _make_batches(6), n_epochs=2,
                           lr=1e-2, device=torch.device("cpu"))
    assert info["steps"] == info["steps_run"] == 6


def test_run_score_arithmetic():
    s = vs.run_score({"SH@10": 0.0035, "ndcg@10": 0.0475}, sh_poison=0.014, ndcg_poison=0.05)
    assert abs(s["removal"] - 0.75) < 1e-9 and abs(s["tau_vs_poison"] - 0.95) < 1e-9
    assert abs(s["score"] - 0.70) < 1e-9 and s["feasible"] == 1.0
    up = vs.run_score({"SH@10": 0.021, "ndcg@10": 0.05}, 0.014, 0.05)
    assert up["removal"] < 0                                   # exposure increase penalized
    tiny = vs.run_score({"SH@10": 0.0, "ndcg@10": 0.05}, 0.0001, 0.05)
    assert abs(tiny["removal"] - 0.0001 / 0.005) < 1e-12       # sh_floor caps the gain


def test_run_level_selection_patience_and_restore():
    m = MockTigerModel()
    # origin, then E=1..: removal grows, utility falls; E=3 is best at lambda=1
    script = [(0.014, 0.05),
              (0.007, 0.0499),   # 0.5   - 0.002 = 0.498
              (0.002, 0.0495),   # 0.857 - 0.010 = 0.847
              (0.0, 0.0490),     # 1.0   - 0.020 = 0.980
              (0.0, 0.0480),     # 1.0   - 0.040 = 0.960
              (0.0, 0.0485)]     # 0.970  -> second miss, stop
    calls = _scripted(m, script)

    def run_once(E):
        with torch.no_grad():
            m.bias_.fill_(float(E))
        return ("kept", E)

    kept, info = vs.run_level_selection(model=m, batches=[], run_once=run_once, seed=2,
                                        patience=2, max_epochs=10)
    assert info["best_n_epochs"] == 3 and kept == ("kept", 3), (info["best_n_epochs"], kept)
    assert [h["n_epochs"] for h in info["history"]] == [1, 2, 3, 4, 5]
    assert float(m.bias_) == 3.0 and len(calls) == 6


def test_run_level_selection_tau_floor():
    m = MockTigerModel()
    _scripted(m, [(0.014, 0.05), (0.004, 0.0490), (0.0, 0.0470)])   # E=2 below 0.95
    kept, info = vs.run_level_selection(model=m, batches=[], run_once=lambda E: E,
                                        seed=2, patience=5, max_epochs=2)
    assert info["best_n_epochs"] == 1 and kept == 1


def test_run_level_score_tol_replays_seed2_mid():
    """beauty/mid seed 2 (job 11112668): without a tolerance E=5 beats E=3 by 4e-5."""
    hist = [(0.004650746937841177, 0.05002771317958832), (0.0006260620430111885, 0.04994826018810272),
            (0.0001341561583103612, 0.04991753399372101), (4.471871943678707e-05, 0.04930567368865013),
            (0.0, 0.04944155365228653), (0.0, 0.04885321110486984), (0.0, 0.048369795083999634)]
    origin = (0.01399695873260498, 0.04988833889365196)
    for tol, want, n_run in ((0.0, 5, 7), (0.01, 3, 5)):
        m = MockTigerModel()
        _scripted(m, [origin] + hist)
        _, info = vs.run_level_selection(model=m, batches=[], run_once=lambda E: E, seed=2,
                                         patience=2, max_epochs=12, score_tol=tol)
        assert info["best_n_epochs"] == want, (tol, info["best_n_epochs"])
        assert len(info["history"]) == n_run, (tol, len(info["history"]))


def test_request_input_cache():
    """_unified_request_inputs: prepare once per request dir, deep copies, RNG restored."""
    from src.models.modules.semantic_id.tiger_unlearning_module import TigerUnlearningModule

    class Fake:
        _unified_request_inputs = TigerUnlearningModule._unified_request_inputs
        clear_request_cache = TigerUnlearningModule.clear_request_cache

        def __init__(self):
            self.n_prep = 0

        def _prepare_unlearning_context(self, **kw):
            self.n_prep += 1
            torch.rand(5)                                   # prep consumes RNG
            return {"forget_batches": [torch.arange(3.0)], "visible_forget_items": {1},
                    "meta": {"target_items": [1]}}

        def _build_coherence_neighbors(self, **kw):
            return [torch.ones(2)]

        def _resolve_code_row_keep(self, ctx, cfg, **kw):
            return None

    f = Fake()
    cfg = {"_cache_request_context": True, "lambda_n": 0.01}
    a, coh_a, _ = f._unified_request_inputs(cfg, 0.01, data_dir="req/0")
    draw_a = torch.rand(3)
    a["forget_batches"][0].add_(100.0)                      # a pass mutating its batch
    b, coh_b, _ = f._unified_request_inputs(cfg, 0.01, data_dir="req/0")
    draw_b = torch.rand(3)
    assert f.n_prep == 1                                    # prepared once
    assert torch.equal(b["forget_batches"][0], torch.arange(3.0))   # untouched copy
    assert torch.equal(draw_a, draw_b)                      # same RNG after prep
    f._unified_request_inputs(cfg, 0.01, data_dir="req/1")
    assert f.n_prep == 2                                    # new request -> prepared
    try:                                                    # flag without a key fails loudly
        f._unified_request_inputs(cfg, 0.01, output_dir="req/0")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    f.clear_request_cache()
    f._unified_request_inputs({"lambda_n": 0.01}, 0.01, data_dir="req/0")
    f._unified_request_inputs({"lambda_n": 0.01}, 0.01, data_dir="req/0")
    assert f.n_prep == 4                                    # no flag -> no caching


def main() -> int:
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
