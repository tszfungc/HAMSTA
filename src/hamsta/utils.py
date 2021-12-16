from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import grad, jit
from scipy import stats

jax.config.update("jax_enable_x64", True)


def SVD(A_mat, Q, outprefix=None):
    # read rfmix output

    # A_mat, Q = io.read_rfmix(fprefix)

    A_std = stats.zscore(A_mat, axis=1)

    print("Project out global ancestry", flush=True)
    P = np.eye(Q.shape[0]) - np.outer(Q, Q) / np.sum(Q ** 2)
    AtP = A_std @ P

    SDpj = np.sqrt(np.sum(AtP ** 2, axis=1))

    print("Running SVD", flush=True)

    U, S, Vt = jnp.linalg.svd(AtP, full_matrices=False)

    if outprefix is not None:
        np.save(outprefix + ".SVD.U.npy", U)
        np.save(outprefix + ".SVD.S.npy", S)
        np.save(outprefix + ".SVD.SDpj.npy", SDpj)
        print("SVD out saved to " + outprefix + ".SVD.*.npy")

    print("Finish SVD", flush=True)

    return U, S, SDpj


def rotate_Z(U, SDpj, Z, Rsq_Q=0.0):

    rotated_Z = np.sqrt(1 - Rsq_Q) * (U.T * SDpj @ Z)

    return jnp.array(rotated_Z)


# Minimize wrapper
# https://gist.github.com/slinderman/24552af1bdbb6cb033bfea9b2dc4ecfd
def minimize(
    fun: Callable,
    x0: jnp.ndarray,
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
):

    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
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


if __name__ == "__main__":
    # CLI for debug only
    pass
