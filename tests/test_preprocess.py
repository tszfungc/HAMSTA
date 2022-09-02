import jax.numpy as jnp
import numpy as np

from hamsta.preprocess import SVD


def test_SVD():
    X = jnp.array(np.random.normal(size=(1000, 100)))
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    U, S = SVD(X)
    assert np.allclose(U * S ** 2 @ U.T, X @ X.T, atol=1e-1)

    tU, tS = SVD(X, k=X.shape[1])
    assert np.allclose(tU * tS ** 2 @ tU.T, X @ X.T, atol=1e-1)

    Q = jnp.array(np.random.normal(size=(100,)))
    pU, pS = SVD(X, Q=Q)
    P = jnp.eye(Q.shape[0]) - jnp.outer(Q, Q) / jnp.sum(Q ** 2)

    assert np.allclose(pU * pS ** 2 @ pU.T, X @ P @ X.T, atol=1e-1)
