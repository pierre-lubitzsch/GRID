"""One optimizer factory for every unlearning algorithm.

Before this, the optimizer was selectable for `unified`, `finetune` and `tracer`
but hardcoded to Adam inside `seif`, `kookmin`, `fanchuan` and `neg_train`, so an
"optimizer" axis in an experiment grid silently did nothing for half the
algorithms. This centralises the choice so `adam | adamw | sgd` means the same
thing everywhere, and so an unknown name fails loudly instead of being ignored.

`scif` deliberately has no entry: it is a Newton/conjugate-gradient step, not a
first-order loop, so there is no optimizer to choose.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

import torch

# SGD gets momentum by default because the algorithms that used to hardcode Adam
# were tuned with an adaptive method; plain SGD without momentum is a much weaker
# drop-in replacement and would make the axis look worse than it is.
SGD_DEFAULT_MOMENTUM = 0.9

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

ParamsLike = Union[Iterable[torch.nn.Parameter], List[Dict[str, Any]]]


def available_optimizers() -> List[str]:
    return sorted(_OPTIMIZERS)


def build_optimizer(
    name: str,
    params: ParamsLike,
    lr: float,
    *,
    weight_decay: float = 0.0,
    momentum: float = SGD_DEFAULT_MOMENTUM,
    algo: str = "",
) -> torch.optim.Optimizer:
    """Build ``name`` over ``params`` (a param iterable OR param groups).

    Only kwargs the chosen optimizer actually accepts are passed: handing
    ``momentum`` to Adam raises, and handing ``betas`` to SGD raises, so the
    caller should not have to branch.
    """
    key = str(name).strip().lower()
    if key not in _OPTIMIZERS:
        raise ValueError(
            f"{algo or 'unlearning'} optimizer must be one of "
            f"{available_optimizers()}, got {name!r}"
        )
    kwargs: Dict[str, Any] = {"lr": float(lr)}
    if float(weight_decay) != 0.0:
        kwargs["weight_decay"] = float(weight_decay)
    if key == "sgd":
        kwargs["momentum"] = float(momentum)
    return _OPTIMIZERS[key](params, **kwargs)
