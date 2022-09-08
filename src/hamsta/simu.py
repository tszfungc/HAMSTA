from functools import partial

import jax.numpy as jnp
from jax import jit, random
import jax


def simu_pheno(
    A: jnp.ndarray,
    Q: jnp.ndarray = None,
    cov: jnp.ndarray = None,
    pve: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
    rep: int = 1,
) -> jnp.ndarray:
    """Simulate phenotypes

    Assume Q and cov are column vector for now

    Args:
        A: local ancestry matrix (marker, sample)
        Q: global ancestry matrix (sample, 1)
        cov: covariate matrix (sample, 1)
        pve: phenotypic variances explained by ``A``, ``Q`` and ``cov``
        rep: number of replicates
    """

    A = (A - A.mean(axis=1, keepdims=True)) / A.std(axis=1, keepdims=True)

    if Q is not None:
        Q = (Q - Q.mean(axis=0, keepdims=True)) / Q.std(axis=0, keepdims=True)
        Q = jnp.repeat(Q, rep, axis=1)
    else:
        Q = jnp.zeros(shape=(A.shape[1], rep))

    if cov is not None:
        cov = (cov - cov.mean(axis=0, keepdims=True)) / cov.std(axis=0, keepdims=True)
        cov = jnp.repeat(cov, rep, axis=1)
    else:
        cov = jnp.zeros(shape=(A.shape[1], rep))

    main_key = random.PRNGKey(12)
    subkey0, subkey1, main_key = random.split(main_key, num=3)

    b = random.normal(key=subkey0, shape=(A.shape[0], rep))
    Ab = A.T @ b
    Ab /= Ab.std(axis=0)

    e_ = random.normal(key=subkey1, shape=(A.shape[1], rep))

    pve_rt = jnp.sqrt(pve)
    pheno = (
        Ab * pve_rt[0] + Q * pve_rt[1] + cov * pve_rt[2] + e_ * jnp.sqrt(1 - pve.sum())
    )

    return pheno


def assoc(
    y: jnp.ndarray,
    x: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """Compute T statistics

    args:
        y: y (sample, rep)
        x: x (sample, marker)
        c: covariates (sample, ncov)

    returns:
        t-statistics (marker, rep)
    """

    C = jnp.hstack([c, jnp.ones(shape=(c.shape[0], 1))])
    P = jnp.eye(C.shape[0]) - C @ jnp.linalg.solve(C.T @ C, C.T)

    Y = P @ y
    # X = P @ x

    scan = jax.vmap(_assoc_single, in_axes=(None, 1, None, None), out_axes=0)
    tstat = scan(Y, x, P, C.shape[1])
    # # marginal beta estimates, (marker, replicates)
    # beta_hat = jnp.multiply(1 / jnp.sum(X.T ** 2, axis=1, keepdims=True), X.T @ Y)

    # # for each marker, compute r col vecs of fitted phenotype
    # fitted = jnp.einsum("sm,mr->msr", X, beta_hat)  # (m,s,r)
    # s2 = jnp.var(
    #     Y - fitted, axis=1, ddof=C.shape[1], keepdims=True
    # )  # var[(s, r) - (m,s,r)] -> (m, r)
    # se = jnp.sqrt(
    #     s2 / jnp.sum(X.T ** 2, axis=1, keepdims=True)
    # )  # (m, r) / (m, 1) -> (m, r)

    # tstat = beta_hat / se  # (m, r) / (m, r) -> (m, r)

    return tstat


@partial(jit, static_argnames=('ddof',))
def _assoc_single(
    Y: jnp.ndarray,
    X: jnp.ndarray,
    P: jnp.ndarray,
    ddof=1,
) -> jnp.ndarray:
    """Compute test statistics for one marker over replicates

    args:
        Y: residualized Y (sample, rep)
        X: Column vector of one marker (sample, )
        P: Residual marker (sample, sample)

    returns:
        vector of t statistics (1, rep)

    """
    X = P @ X  # (sample, )
    beta_hat = jnp.multiply(
        1 / jnp.sum(X.T ** 2, keepdims=True), X.T @ Y
    )  # (rep, )
    fitted = jnp.outer(X, beta_hat)  # (sample, rep)

    s2 = jnp.var(Y - fitted, axis=0, ddof=ddof)  # (rep, )
    se = jnp.sqrt(s2 / jnp.sum(X.T ** 2 , keepdims=True))  # (rep, )

    tstat = beta_hat / se # (rep, )

    return tstat


def assoc_null(
    y: jnp.ndarray,
    x: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """Compute T statistics

    args:
        y: y (sample, rep)
        x: x (sample, marker)
        c: covariates (sample, ncov)

    """

    C = jnp.hstack([c, jnp.ones(shape=(c.shape[0], 1))])
    P = jnp.eye(C.shape[0]) - C @ jnp.linalg.solve(C.T @ C, C.T)

    Y = P @ y
    X = P @ x

    # marginal beta estimates, (marker, replicates)
    beta_hat = jnp.multiply(1 / jnp.sum(X.T ** 2, axis=1, keepdims=True), X.T @ Y)

    # for each marker, compute r col vecs of fitted phenotype
    fitted = jnp.einsum("sm,mr->msr", X, beta_hat)  # (m,s,r)
    s2 = jnp.ones(
        shape=(fitted.shape[0], fitted.shape[2])
    )  # var[(s, r) - (m,s,r)] -> (m, r)
    se = jnp.sqrt(
        s2 / jnp.sum(X.T ** 2, axis=1, keepdims=True)
    )  # (m, r) / (m, 1) -> (m, r)

    tstat = beta_hat / se  # (m, r) / (m, r) -> (m, r)

    return tstat
