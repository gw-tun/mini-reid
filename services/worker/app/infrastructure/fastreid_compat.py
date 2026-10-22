"""Runtime shims that let FastReID v0.1.1 import on modern Python and torch.

FastReID v0.1.1 predates Python 3.10 and torch 1.9. Two imports in its
`fastreid.data.build` and `fastreid.engine.defaults` modules were removed from
the standard library / torch:

* `from collections import Mapping` -> `collections.abc.Mapping`
* `from torch._six import string_classes, int_classes, container_abcs`

Importing this module installs both replacements in ``sys.modules`` and on the
``collections`` / ``torch`` modules *before* ``fastreid`` is imported. It is a
no-op on versions that no longer need it.
"""

from __future__ import annotations

import collections
import collections.abc
import sys
import types

_ABC_NAMES = (
    "Mapping",
    "MutableMapping",
    "Sequence",
    "MutableSequence",
    "Iterable",
    "Iterator",
    "Callable",
    "Set",
    "MutableSet",
    "Hashable",
    "ItemsView",
    "KeysView",
    "ValuesView",
    "Container",
    "Sized",
    "Generator",
    "ByteString",
)


def install_collections_abc() -> None:
    for name in _ABC_NAMES:
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name, None))


def install_torch_six() -> None:
    try:
        import torch
    except ImportError:
        return
    if "torch._six" in sys.modules:
        return
    module = types.ModuleType("torch._six")
    module.string_classes = (str,)
    module.int_classes = int
    module.container_abcs = collections.abc
    sys.modules["torch._six"] = module
    torch._six = module


def install() -> None:
    install_collections_abc()
    install_torch_six()


install()
