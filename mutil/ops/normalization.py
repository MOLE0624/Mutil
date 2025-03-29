# ================================================================================
# File       : normalizatioin.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a normalization operation
# Date       : 2025-03-29
# ================================================================================

from enum import Enum

import jax
import jax.numpy as jnp


class NormMethod(Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    COMPOUND = "compound"


@jax.jit
def normalization(
    input_data, method, min_vals, max_vals, mean_vals=None, std_vals=None
):
    min_vals = jnp.array(min_vals)
    max_vals = jnp.array(max_vals)

    if method == NormMethod.MINMAX:
        # Broadcasting optimization
        return (input_data - min_vals[None, :, None, None]) / (
            max_vals[None, :, None, None] - min_vals[None, :, None, None]
        )
    elif method == NormMethod.ZSCORE:
        mean_vals = jnp.array(mean_vals)
        std_vals = jnp.array(std_vals)
        # Broadcasting optimization
        return (input_data - mean_vals[None, :, None, None]) / std_vals[
            None, :, None, None
        ]
    elif method == NormMethod.COMPOUND:
        minmax = normalization(input_data, NormMethod.MINMAX, min_vals, max_vals)
        # Broadcasting optimization
        return (minmax - mean_vals[None, :, None, None]) / std_vals[None, :, None, None]
    else:
        raise ValueError("Unknown normalization method")
