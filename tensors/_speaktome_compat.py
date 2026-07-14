"""Back-compat shims bridging speaktome's older AbstractTensor call sites
onto the vendored (turing-lineage) tensor abstraction.

This file exists so the vendored ``abstraction.py`` can stay an unmodified,
diffable copy of the upstream source (turing/nodus lineage) while
speaktome's own consumer code -- written against an older, pre-rename API
-- keeps working. Add new aliases here as gaps are discovered; do not
patch them directly into ``abstraction.py``.
"""
from __future__ import annotations

from .abstraction import AbstractTensor, AbstractScalar, BACKEND_REGISTRY
# --- END HEADER ---


# ---------------------------------------------------------------------
# Upstream bug workaround (not a speaktome-specific rename)
# ---------------------------------------------------------------------
# AbstractScalar.__new__ dynamically re-classes an already-built tensor
# (mutates tensor.__class__ to a hybrid backend+AbstractScalar type) and
# returns that same, already-initialized instance. Because the returned
# object's type is now a subclass of AbstractScalar, Python's normal
# object-construction protocol then calls __init__ on it *again* with the
# original call args -- e.g. ``AbstractScalar(result)`` re-invokes
# ``PyTorchTensorOperations.__init__(result_instance, result)`` (found
# first in the dynamic class's MRO, ahead of AbstractScalar itself),
# feeding the wrapped tensor in as if it were ``default_device``. This
# breaks any reduction that returns a 0-d scalar (``max()``, ``min()``,
# etc.) on the torch backend. Overriding ``AbstractScalar.__init__``
# alone doesn't fix it -- the backend class's own __init__ shadows it in
# the MRO -- so the dynamically created hybrid class needs its *own*
# no-op __init__, taking precedence over both parents.
_scalar_cache: dict = {}


def _abstract_scalar_new(cls, tensor):
    if getattr(getattr(tensor, "data", None), "shape", ()) != ():
        raise ValueError("AbstractScalar requires a zero-dimensional tensor")
    if isinstance(tensor, AbstractScalar):
        return tensor
    base = tensor.__class__
    scalar_cls = _scalar_cache.get(base)
    if scalar_cls is None:
        scalar_cls = type(
            f"{base.__name__}Scalar",
            (base, cls),
            {"__init__": lambda self, *args, **kwargs: None},
        )
        _scalar_cache[base] = scalar_cls
    tensor.__class__ = scalar_cls
    return tensor


AbstractScalar.__new__ = staticmethod(_abstract_scalar_new)


def _check_or_build_registry_compat():
    """Prefer torch when picking a default backend for a bare ``get_tensor()``.

    Upstream's registry order is ``("numpy", "torch", "pure_python")`` --
    numpy wins whenever it's installed, even if torch is also available.
    Several numpy-backend cast methods in this vendored snapshot
    (``long_cast_``, ``double_``, ``int_``, ``bool_``) have a pre-existing
    signature mismatch with their shared caller and are currently broken
    (see ``AGENTS`` notes / speaktome vision brief for context). Speaktome
    itself declares ``FACULTY_REQUIREMENT = Faculty.TORCH`` throughout
    (the GPT-2 scorer requires torch anyway), so prefer torch first here
    and fall back to numpy/pure_python only if torch truly isn't
    available -- rather than deep-repairing the numpy backend, which is
    out of scope for this compatibility bridge.
    """
    cls = None
    if not BACKEND_REGISTRY:
        try:
            from . import torch_backend  # noqa: F401
        except Exception:
            pass
        try:
            from . import numpy_backend  # noqa: F401
        except Exception:
            pass
        try:
            from . import pure_backend  # noqa: F401
        except Exception:
            pass

    for backend_name in ("torch", "numpy", "pure_python"):
        backend_cls = BACKEND_REGISTRY.get(backend_name)
        if backend_cls is not None:
            cls = backend_cls
            break
    if cls is None:
        raise RuntimeError("No tensor backend available for tensor creation.")
    return cls


AbstractTensor.check_or_build_registry = staticmethod(_check_or_build_registry_compat)


def _tensor_from_list_compat(cls, data, dtype=None, device=None):
    """Alias for :meth:`AbstractTensor._tensor_from_list`.

    Speaktome's ``speaktome/core/*.py`` and its tests call
    ``tensor_ops.tensor_from_list(data, dtype=..., device=...)`` (no
    leading underscore, the pre-rename name). The upstream lineage renamed
    the public entry point to ``_tensor_from_list``. Works both as a
    classmethod call (``AbstractTensor.tensor_from_list(...)``) and via an
    instance (``ops.tensor_from_list(...)``, which auto-binds ``cls`` to
    the instance's own backend class).
    """
    return cls._tensor_from_list(data, dtype=dtype, device=device)


if not hasattr(AbstractTensor, "tensor_from_list"):
    AbstractTensor.tensor_from_list = classmethod(_tensor_from_list_compat)


class _CallableShape(tuple):
    """A tuple that is also callable, returning itself.

    Speaktome's ``speaktome/core/*.py`` (e.g. ``lookahead_controller.py``,
    ``beam_search.py``) calls ``tensor.shape()`` -- the pre-rename
    convention where ``shape`` was a bound method. The upstream lineage
    turned it into a plain ``@property`` returning a tuple directly. This
    wrapper supports both: ``t.shape[0]`` and ``t.shape()[0]`` both work.
    """

    def __call__(self):
        return self


def _shape_compat(self):
    return _CallableShape(self.shape_())


AbstractTensor.shape = property(_shape_compat)


# ---------------------------------------------------------------------
# The "ops as generic dispatcher" pattern
# ---------------------------------------------------------------------
# Speaktome's core code (LookaheadController, BeamSearch, ...) holds one
# "empty" AbstractTensor instance (self.tensor_ops, from
# ``AbstractTensor.get_tensor()`` with no data) and calls
# ``self.tensor_ops.METHOD(target_tensor, *rest)`` for a whole family of
# methods -- treating tensor_ops as a stateless namespace of functions
# rather than binding the method to the data it operates on. The upstream
# lineage bound these methods to the tensor itself
# (``target_tensor.METHOD(*rest)``), which is the correct direction to go,
# but breaks every one of speaktome's existing call sites at once.
#
# Rather than hand-patch a dozen individual methods, redirect generically:
# if the first positional argument to one of these methods is itself an
# AbstractTensor, treat it as "the real target" and call the method on it
# instead of on ``self``. This assumes ``self`` in these calls is always
# the throwaway dispatcher object, never a real tensor meaningfully
# combined with another tensor via this exact call syntax -- true for
# speaktome's current call sites, but worth knowing if this file is ever
# extended.
_REDIRECT_IF_FIRST_ARG_IS_TENSOR = [
    "get_dtype",
    "max",
    "to_device",
    "clone",
    "long_cast",
    "not_equal",
    "clamp",
    "select_by_indices",
    "repeat_interleave",
    "view_flat",
    "less",
    "boolean_mask_select",
    "assign_at_indices",
    "increment_at_indices",
]


def _make_redirecting_method(name: str):
    upstream = getattr(AbstractTensor, name)

    def _redirecting_method(self, *args, **kwargs):
        if args and isinstance(args[0], AbstractTensor):
            target, rest = args[0], args[1:]
            return getattr(target, name)(*rest, **kwargs)
        return upstream(self, *args, **kwargs)

    _redirecting_method.__name__ = name
    _redirecting_method.__doc__ = (
        f"Back-compat wrapper: if called as ops.{name}(tensor, ...), "
        f"redirect to tensor.{name}(...). See module docstring."
    )
    return _redirecting_method


for _name in _REDIRECT_IF_FIRST_ARG_IS_TENSOR:
    setattr(AbstractTensor, _name, _make_redirecting_method(_name))


_item_upstream = AbstractTensor.item


def _item_compat(self, value=None):
    """Extend :meth:`AbstractTensor.item` to accept any scalar-like value.

    Speaktome calls ``some_ops.item(x)`` where ``x`` may be an
    ``AbstractTensor``, a raw backend scalar (e.g. a ``numpy.int64`` from
    indexing a 1-D tensor with a single int, which ``__getitem__`` returns
    unwrapped), or already a plain Python number.
    """
    if value is None:
        return _item_upstream(self)
    if isinstance(value, AbstractTensor):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


AbstractTensor.item = _item_compat
