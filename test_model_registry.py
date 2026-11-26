"""The model axis: Python registry and shell resolver must not drift apart.

There are two sources of truth for "what does MODEL=<name> mean" -- src/models/
registry.py (used in Python) and scripts/resolve_model.sh (used by the sbatch
runners). If they disagree, a run trains one architecture and is evaluated as
another, which is the kind of thing that shows up as an unexplained metric shift
months later. This pins them together.

    python test_model_registry.py
"""

import os
import re
import subprocess
import sys

from src.models.registry import available_models, resolve_model

ROOT = os.path.dirname(os.path.abspath(__file__))
PASS = []

KEYS = {
    "GRID_TRAIN_EXPERIMENT": "train_experiment",
    "GRID_INFER_EXPERIMENT": "inference_experiment",
    "GRID_UNLEARN_EXPERIMENT": "unlearn_experiment",
    "GRID_MODEL_TARGET": "target",
    "GRID_MODEL_TAG": "run_tag_prefix",
}


def shell_resolve(name: str):
    """Run the bash resolver in a clean env and read back what it exported."""
    script = (
        f'source "{ROOT}/scripts/resolve_model.sh" && resolve_model "{name}" && '
        + " && ".join(f'echo "{k}=${{{k}}}"' for k in KEYS)
    )
    # env -i would drop PATH; instead clear only the GRID_* vars the resolver
    # honours as overrides, or a previous iteration's value sticks (the resolver
    # uses ${VAR:-default} on purpose so callers can override it).
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"resolver failed for {name!r}: {p.stderr.strip()}")
    return dict(
        line.split("=", 1) for line in p.stdout.strip().splitlines() if "=" in line
    )


for name in available_models():
    spec = resolve_model(name)
    got = shell_resolve(name)
    for env_key, attr in KEYS.items():
        want = getattr(spec, attr)
        assert got.get(env_key, "") == want, (
            f"{name}: {env_key} is {got.get(env_key)!r} in the shell resolver but "
            f"{want!r} in the Python registry"
        )
PASS.append(f"1. registry and resolver agree on all {len(available_models())} models "
            f"({', '.join(available_models())})")

# tiger MUST keep an empty tag: every recorded run dir, extractor regex and
# results table predates this axis and keys off tags with no model token.
assert resolve_model("tiger").run_tag_prefix == "", \
    "giving tiger a run-tag prefix orphans every recorded result"
assert resolve_model("TIGER").name == "tiger", "resolution must be case-insensitive"
assert resolve_model("").name == "tiger", "empty name must default to tiger"
PASS.append("2. tiger keeps an empty run-tag prefix; resolution is case-insensitive")

for bad in ("nope", "tigerr"):
    try:
        resolve_model(bad)
        raise AssertionError(f"Python registry accepted unknown model {bad!r}")
    except KeyError:
        pass
    try:
        shell_resolve(bad)
        raise AssertionError(f"shell resolver accepted unknown model {bad!r}")
    except RuntimeError:
        pass
PASS.append("3. unknown models are rejected by BOTH, not silently defaulted")

missing = []
for name in available_models():
    spec = resolve_model(name)
    for exp in (spec.train_experiment, spec.inference_experiment,
                spec.unlearn_experiment):
        path = os.path.join(ROOT, "configs", "experiment", f"{exp}.yaml")
        if not os.path.exists(path):
            missing.append(f"{name}: configs/experiment/{exp}.yaml")
assert not missing, "registry names configs that do not exist:\n  " + "\n  ".join(missing)
PASS.append("4. every experiment config named by the registry exists on disk")

# The differentiable-ids flag drives whether unlearning must re-commit the SID
# tensor afterwards, so it must be right.
assert resolve_model("tiger").differentiable_ids is False
assert resolve_model("diger").differentiable_ids is True
# letter's tokenizer is LEARNABLE but not JOINT: it is trained in its own stage
# and its ids are frozen before the recommender starts, so an unlearning update
# cannot move them and there is nothing to re-commit.
assert resolve_model("letter").differentiable_ids is False
PASS.append("5. differentiable_ids is set correctly per model")

# --- run-dir naming conflicts -----------------------------------------------
# Adding a model changes how run dirs are named, and dataset-keyed globs absorb
# the model tag into their leading wildcard. These pin that the exclusion helper
# exists, stays in sync with the registry, and actually works.
import tempfile
import glob as _glob

nd_py = sorted(n for n in available_models() if resolve_model(n).run_tag_prefix)
p = subprocess.run(
    ["bash", "-c",
     f'source "{ROOT}/scripts/resolve_model.sh" && '
     'echo "$GRID_MODEL_TAGS_NON_DEFAULT" && grid_other_model_regex'],
    capture_output=True, text=True,
)
assert p.returncode == 0, p.stderr
nd_sh = sorted(p.stdout.splitlines()[0].split())
regex = p.stdout.splitlines()[1].strip()
assert nd_sh == nd_py, (
    f"GRID_MODEL_TAGS_NON_DEFAULT is {nd_sh} but the registry's non-default "
    f"models are {nd_py}; a dataset-keyed glob would silently pick up the "
    "missing model's runs as tiger's"
)
PASS.append(f"6. non-default model list matches the registry ({', '.join(nd_py)})")

with tempfile.TemporaryDirectory() as td:
    made = []
    for lab in ("12-00_job1_beauty_clean", "12-00_job2_diger_beauty_clean",
                "12-00_job3_beauty_poison_pct1_n1_tgtmid",
                "12-00_job4_diger_beauty_poison_pct1_n1_tgtmid",
                "12-00_job5_letter_beauty_clean",
                "12-00_job6_letter_beauty_poison_pct1_n1_tgtmid"):
        d = os.path.join(td, "logs/train/runs/d", lab, "checkpoints")
        os.makedirs(d)
        open(os.path.join(d, "checkpoint_epoch=000_step=001.ckpt"), "w").close()
        made.append(lab)
    raw = _glob.glob(os.path.join(td, "logs/train/runs/*/*_beauty_clean"))
    assert len(raw) == 3, "test setup: the unfiltered glob should catch all three"
    keep = [d for d in raw if not re.search(regex, os.path.basename(d))]
    assert len(keep) == 1 and not any(m in keep[0] for m in nd_py), \
        f"exclusion regex {regex!r} did not filter the non-tiger runs: {keep}"
PASS.append(f"7. the exclusion regex removes {'/'.join(nd_py)} runs from a "
            "tiger-targeted glob")

sys.path.insert(0, ROOT)
from scripts.extract_model_comparison import find_runs  # noqa: E402
_orig = os.getcwd()
try:
    os.chdir(ROOT)
    t_clean = {os.path.basename(d) for d in find_runs("beauty", "", "clean")}
    other = {
        n: {os.path.basename(d) for d in find_runs("beauty", f"{n}_", "clean")}
        for n in nd_py
    }
finally:
    os.chdir(_orig)
for _n, _dirs in other.items():
    assert not (t_clean & _dirs), \
        f"find_runs returned the same dir for tiger AND {_n}"
    assert not any(f"_{_n}_" in b for b in t_clean), \
        f"tiger lookup picked up {_n} runs: {[b for b in t_clean if f'_{_n}_' in b]}"
PASS.append("8. extract_model_comparison separates tiger "
            f"({len(t_clean)}) from " +
            ", ".join(f"{n} ({len(d)})" for n, d in other.items()) + " clean runs")

# --- per-model config drift ------------------------------------------------
# The non-default models' experiment configs are FULL COPIES of tiger's, not
# overlays, so a knob added to a tiger config does not reach them. That is not
# hypothetical: `lambda_p` was added to tiger_unlearn_scif_flat.yaml on
# 2026-08-31 and the letter copies, generated hours earlier, silently did not
# have it -- under Hydra struct mode `unlearning.lambda_p=...` would then abort
# for MODEL=letter only. Compare the COMPOSED configs, because some of these are
# thin overlays that pick keys up from configs/unlearning_defaults.yaml.
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf                   # noqa: E402

_OV = ["data_dir=/x", "semantic_id_path=/x", "ckpt_path=/x", "num_hierarchies=4"]


def _unlearning_keys(experiment: str) -> set:
    with initialize_config_dir(
        config_dir=os.path.join(ROOT, "configs"), version_base="1.3"
    ):
        cfg = compose("unlearn", overrides=[f"experiment={experiment}"] + _OV)
    return set(OmegaConf.to_container(cfg.unlearning))


drift, checked = [], 0
for _tiger_cfg in sorted(_glob.glob(os.path.join(ROOT, "configs/experiment/tiger_unlearn_*.yaml"))):
    _base = os.path.basename(_tiger_cfg)[:-len(".yaml")]
    _tiger_keys = None
    for _name in available_models():
        _tag = resolve_model(_name).run_tag_prefix
        if not _tag:
            continue
        _other = f"{_name}_{_base[len('tiger_'):]}"
        if not os.path.exists(os.path.join(ROOT, "configs", "experiment", f"{_other}.yaml")):
            # Deliberate gaps are allowed (diger ships only its scif config);
            # grid_model_config fails loudly for those, so they are not silent.
            continue
        if _tiger_keys is None:
            _tiger_keys = _unlearning_keys(_base)
        _missing = sorted(_tiger_keys - _unlearning_keys(_other))
        checked += 1
        if _missing:
            drift.append(f"{_other}: missing {_missing}")
assert not drift, (
    "per-model unlearn configs have drifted from tiger's:\n  " + "\n  ".join(drift)
)
PASS.append(f"9. all {checked} non-tiger unlearn configs expose every knob "
            "tiger's do (no silent per-model drift)")

print("\n".join("PASS  " + p for p in PASS))
print(f"\nAll {len(PASS)} model-registry checks passed.")
