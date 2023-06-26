import logging
from typing import Tuple

import jax.numpy as jnp
from sklearn.utils.extmath import randomized_svd

_logger = logging.getLogger(__name__)


def SVD(
    A: jnp.ndarray,
    Q: jnp.ndarray = None,
    LAD: jnp.ndarray = None,
    k: int = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SVD, with covariate and output options.

    | When k is set, run truncated SVD in scikit learn.
    | Else, compute all the components with jax SVD

    Args:
        A:
            local ancestry matrix (marker, sample)
        Q:
            global ancestry or covariates to be projected (sample, n_covariate)
        LAD:
            local ancestry correlation matrix (marker, marker)
        k:
            number of components computed in truncated

    Returns:
        ``(U, S)`` in SVD of ``X = U * S @ Vh``, where X is A/sqrt(N) with A standardized

    """
    
    if LAD is not None:
        A_std = LAD
    else:
        # Standardize
        _logger.info("Standardize local ancestry")
        A = (A - A.mean(axis=1, keepdims=True)) / A.std(axis=1, keepdims=True)

        # Projection
        if Q is not None:
            _logger.info(f"Start: Project out global ancestry : {Q.shape[0]}")
            if len(Q.shape) == 1:
                Q = Q.reshape(-1, 1)
            Q -= Q.mean(axis=0, keepdims=True)
            P = jnp.eye(Q.shape[0]) - Q @ jnp.linalg.solve(Q.T @ Q, Q.T)
            A_std = jnp.matmul(A, P)
            _logger.info(f"End: Project out global ancestry : {Q.shape[0]}")
        else:
            A_std = A

    # SVD
    _logger.info("Running SVD")
    if k is None:
        U, S, _ = jnp.linalg.svd(A_std, full_matrices=False)
    else:
        U, S, _ = randomized_svd(A_std, n_components=k, random_state=None)

    
    if LAD is not None:
        S = jnp.sqrt(S)
    else:
        S = S / jnp.sqrt(A.shape[1])

    return U, S
