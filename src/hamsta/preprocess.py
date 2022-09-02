import logging
from typing import Tuple

import jax.numpy as jnp
import numpy as np

_logger = logging.getLogger(__name__)


def SVD(
    A: jnp.ndarray,
    Q: jnp.ndarray,
    outprefix: str = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """jax numpy SVD, with covariate and output options.

    Args:
        A:
            | local ancestry matrix (marker, sample)
        Q:
            | global ancestry or covariates to be projected (sample, n_covariate)
        outprefix:
            | outprefix for writing SVD results.
             When set, write to ``{outprefix}.SVD.U.npy`` and ``{outprefix}.SVD.S.npy``


    Returns:
        ``(U, S)`` in SVD of ``X = U * S @ Vh``

    """

    # Standardize
    _logger.info("Standardize local ancestry")
    A = (A - A.mean(axis=1, keepdims=True)) / A.std(axis=1, keepdims=True)

    # Projection
    _logger.info(f"Project out global ancestry : {Q.shape[0]}")
    Q -= Q.mean(axis=0, keepdims=True)
    P = jnp.eye(Q.shape[0]) - Q @ jnp.linalg.solve(Q.T @ Q, Q)
    A_std = jnp.matmul(A, P)

    # SVD
    _logger.info("Running SVD")
    U, S, _ = jnp.linalg.svd(A_std, full_matrices=False)

    # Write

    if outprefix is not None:
        np.save(outprefix + ".SVD.U.npy", U)
        np.save(outprefix + ".SVD.S.npy", S)
        # np.save(outprefix + ".SVD.SDpj.npy", SDpj)
        _logger.info("SVD out saved to " + outprefix + ".SVD.*.npy")
        _logger.info(f"output dimension: U ({U.shape}) S ({S.shape})")

    return U, S
