"""JAX backend placeholder."""

from typing import Any

from .abstraction import AbstractTensorOperations

# --- END HEADER ---

class JAXTensorOperations(AbstractTensorOperations):
    """Stub backend for future JAX-based tensor operations."""

    # ########## STUB: JAXTensorOperations ##########
    # PURPOSE: Provide an optional JAX backend mirroring the
    #          :class:`PyTorchTensorOperations` API.
    # EXPECTED BEHAVIOR: Implement all tensor operations using
    #          ``jax.numpy`` with device management for CPU/GPU/TPU.
    # INPUTS: JAX arrays and standard Python lists.
    # OUTPUTS: JAX arrays or converted Python data structures.
    # KEY ASSUMPTIONS/DEPENDENCIES: Requires the ``jax`` package and
    #          compatible accelerator drivers.
    # TODO:
    #   - Implement each method using ``jax.numpy`` equivalents.
    #   - Handle device placement and data transfer semantics.
    #   - Integrate this class with :func:`get_tensor_operations`.
    # NOTES: This class currently raises ``NotImplementedError`` to
    #        indicate the backend is not yet available.
    # ###############################################################
    def __init__(self, default_device: str = "cpu", track_time: bool = False) -> None:
        super().__init__(track_time=track_time)
        raise NotImplementedError("JAX backend not yet implemented")

    @staticmethod
    def test() -> None:
        """Demonstrate the current stub behaviour."""
        try:
            JAXTensorOperations()
        except NotImplementedError:
            return
        raise AssertionError("JAXTensorOperations should be a stub")
