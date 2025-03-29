# ================================================================================
# File       : normalizatioin.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a normalization operation
# Date       : 2025-03-29
# ================================================================================

from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class NormMethod(Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    COMPOUND = "compound"


@partial(
    jax.jit,
    static_argnums=(1, 2, 3),
)
def _normalization(
    input_data: np.ndarray,
    method: NormMethod,
    min_vals: tuple,  # np.ndarray,
    max_vals: tuple,  # np.ndarray,
    mean_vals=None,
    std_vals=None,
):
    # Convert lists to tuples if they are passed as lists
    input_data = jnp.array(input_data)
    min_vals = jnp.array(min_vals)
    max_vals = jnp.array(max_vals)
    mean_vals = jnp.array(mean_vals) if mean_vals is not None else None
    std_vals = jnp.array(std_vals) if std_vals is not None else None

    if method == NormMethod.MINMAX:
        return (input_data - min_vals[None, :, None, None]) / (
            max_vals[None, :, None, None] - min_vals[None, :, None, None]
        )
    elif method == NormMethod.ZSCORE:
        return (input_data - mean_vals[None, :, None, None]) / std_vals[
            None, :, None, None
        ]
    elif method == NormMethod.COMPOUND:
        # Combine MINMAX and ZSCORE normalization
        minmax = (input_data - min_vals[None, :, None, None]) / (
            max_vals[None, :, None, None] - min_vals[None, :, None, None]
        )
        return (minmax - mean_vals[None, :, None, None]) / std_vals[None, :, None, None]


def normalization(
    input_data: np.ndarray,
    method: NormMethod,
    min_vals: tuple,
    max_vals: tuple,
    mean_vals=None,
    std_vals=None,
):
    return _normalization(
        input_data, method, min_vals, max_vals, mean_vals, std_vals
    ).block_until_ready()
