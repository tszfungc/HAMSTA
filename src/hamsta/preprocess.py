import logging
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from sklearn.utils.extmath import randomized_svd

_logger = logging.getLogger(__name__)


def SVD(
    A: jnp.ndarray,
    Q: jnp.ndarray = None,
    k: int = None,
    outprefix: str = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SVD, with covariate and output options.

    | When k is set, run truncated SVD in scikit learn.
    | Else, compute all the components with jax SVD

    Args:
        A:
            local ancestry matrix (marker, sample)
        Q:
            global ancestry or covariates to be projected (sample, n_covariate)
        k:
            number of components computed in truncated
        outprefix:
            outprefix for writing SVD results.
            When set, write to ``{outprefix}.SVD.U.npy`` and ``{outprefix}.SVD.S.npy``


    Returns:
        ``(U, S)`` in SVD of ``X = U * S @ Vh``

    """

    # Standardize
    _logger.info("Standardize local ancestry")
    A = (A - A.mean(axis=1, keepdims=True)) / A.std(axis=1, keepdims=True)

    # Projection
    if Q is not None:
        _logger.info(f"Project out global ancestry : {Q.shape[0]}")
        Q = Q.reshape(-1, 1)
        Q -= Q.mean(axis=0, keepdims=True)
        P = jnp.eye(Q.shape[0]) - Q @ jnp.linalg.solve(Q.T @ Q, Q.T)
        A_std = jnp.matmul(A, P)
    else:
        A_std = A

    # SVD
    _logger.info("Running SVD")
    if k is None:
        U, S, _ = jnp.linalg.svd(A_std, full_matrices=False)
    else:
        U, S, _ = randomized_svd(A_std, n_components=k, random_state=None)

    # Write

    if outprefix is not None:
        np.save(outprefix + ".SVD.U.npy", U)
        np.save(outprefix + ".SVD.S.npy", S)
        # np.save(outprefix + ".SVD.SDpj.npy", SDpj)
        _logger.info("SVD out saved to " + outprefix + ".SVD.*.npy")
        _logger.info(f"output dimension: U ({U.shape}) S ({S.shape})")

    return U, S
