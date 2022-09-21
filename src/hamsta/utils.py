from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from jax import grad, jit
from scipy import stats

from hamsta import io

jax.config.update("jax_enable_x64", True)


def make_intercept_design(Z_dim, binsize):
    bin_idx = np.arange(Z_dim) // binsize
    # group the last incomplete to the previous bin
    if Z_dim % binsize != 0:
        bin_idx[bin_idx == max(bin_idx)] -= 1
    bin_idx_design_mat = pd.get_dummies(bin_idx).values

    bin_idx_design_mat = jnp.array(bin_idx_design_mat)
    return bin_idx_design_mat


def SVD(A_mat, Q, outprefix=None):
    # read rfmix output
    A_mat = stats.zscore(A_mat, axis=1)

    print("Project out global ancestry", flush=True)
    Q -= np.mean(Q)
    P = np.eye(Q.shape[0]) - np.outer(Q, Q) / np.sum(Q ** 2)
    A_std = A_mat.astype(np.float) @ P.astype(np.float)

    SDpj = np.sqrt(np.sum(np.multiply(A_std, A_std), axis=1))

    print("Running SVD", flush=True)
    U, S, Vt = jnp.linalg.svd(A_std, full_matrices=False)

    if outprefix is not None:
        np.save(outprefix + ".SVD.U.npy", U)
        np.save(outprefix + ".SVD.S.npy", S)
        np.save(outprefix + ".SVD.SDpj.npy", SDpj)
        print("SVD out saved to " + outprefix + ".SVD.*.npy")
        print(f"output dimension: U ({U.shape}) S ({S.shape})")

    print("Finish SVD", flush=True)

    return U, S


def PCA(LADmat, n_indiv, outprefix=None):
    U, S2, _ = jnp.linalg.svd(LADmat)
    S = jnp.sqrt(S2)

    if outprefix is not None:
        np.save(outprefix + ".SVD.U.npy", U)
        np.save(outprefix + ".SVD.S.npy", S)
        np.save(outprefix + ".SVD.SDpj.npy", np.repeat(np.sqrt(n_indiv), U.shape[0]))
        print("SVD out saved to " + outprefix + ".SVD.*.npy")
        print(f"output dimension: U ({U.shape}) S ({S.shape})")

    print("Finish SVD", flush=True)

    return U, S


def rotate_Z(svdprefix, Z, n_indiv, multichrom=False, n_S=500, yvar=1.0):

    if multichrom:
        rotated_Z = []
        n_variant_cnt = 0
        for i in range(1, 23):
            U, _, SDpj = io.read_SVD_chr(svdprefix, i)
            Z_range = slice(n_variant_cnt, n_variant_cnt + U.shape[0])
            # import pdb; pdb.set_trace()
            rotated_Z.append(np.sqrt(yvar) * U.T @ (SDpj * Z[Z_range]))
            n_variant_cnt += U.shape[0]
        rotated_Z = np.concatenate(rotated_Z)
    else:
        U, _, SDpj = io.read_SVD(svdprefix)
        rotated_Z = np.sqrt(yvar) * U.T[:n_S] @ (SDpj * Z)

    return jnp.array(rotated_Z)


# Minimize wrapper
def minimize(
    fun: Callable,
    x0: jnp.ndarray,
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
    **kwargs,
):

    if options is None:
        options = {}

    fun_with_args = fun

    f_jac = jit(jax.value_and_grad(fun_with_args))

    # Hessian vector product
    def hvp(primals, tangents):
        return jax.jvp(grad(fun_with_args), (primals,), (tangents,))[1]

    results = scipy.optimize.minimize(
        f_jac, x0, method=method, hessp=hvp, jac=True, tol=tol, options=options
    )

    return results


def compute_residvar(
    Y: jnp.ndarray,
    X: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate residual variance Y | independent variables

    Args:
        Y: phenotype
        X: independent variables
    """

    resid_var = []
    for pheno_idx in range(Y.shape[1]):
        fitted = sm.OLS(
            endog=np.array(Y[:, pheno_idx]), exog=sm.add_constant(np.array(X))
        ).fit()
        resid_var.append(1 - fitted.rsquared)

    resid_var = jnp.array(resid_var)

    return resid_var


def estimate_thres(
    U: jnp.ndarray,
    S: jnp.ndarray,
    var: float = 1.0,
    nrep: int = 2000,
    fwer: float = 0.05,
):
    S_scale = np.random.normal(size=(nrep, S.shape[0])) * S

    D = np.sqrt(np.sum((U * S) ** 2, axis=1))

    Z = np.einsum("i, ij,kj->ki", 1 / D, U, S_scale)

    maxchi2 = np.max(Z ** 2, axis=1)
    cutoff = np.percentile(maxchi2, 100 * (1 - fwer))
    return cutoff


if __name__ == "__main__":
    # CLI for debug only
    pass
