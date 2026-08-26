"""Skip the random weight initialization of modules that are about to be overwritten.

Constructing ``TextToLatentRFDiT`` (766 M params) fills every ``nn.Linear`` /
``nn.Embedding`` with ``kaiming_uniform_`` / ``normal_`` on the CPU, and the very
next statement replaces all 714 tensors with the checkpoint.  That RNG pass costs
~1.8 s for the model and ~0.6 s for the codec, and touches ~3 GB of RSS for
nothing (see docs/experiments/11-load-time.md).

Only the *random* fills are suppressed.  Deterministic ones (``zeros_``,
``ones_``, ``constant_``, ``eye_``, ``dirac_``) are left alone because a module
may use them on a buffer that the checkpoint does not carry.  Every call site
loads the checkpoint with ``strict=True`` afterwards, so a tensor that no
checkpoint entry covers raises instead of silently staying uninitialized.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

_RANDOM_INITS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "orthogonal_",
    "sparse_",
)


def _noop(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return tensor


@contextmanager
def skip_random_init(enabled: bool = True):
    """Turn ``torch.nn.init`` random fills into no-ops for the duration."""
    if not enabled:
        yield
        return
    init = torch.nn.init
    saved = {}
    for name in _RANDOM_INITS:
        fn = getattr(init, name, None)
        if fn is None:
            continue
        saved[name] = fn
        setattr(init, name, _noop)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(init, name, fn)
