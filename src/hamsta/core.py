from functools import partial
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import grad, jit
from scipy import stats


# Minimize wrapper
def _minimize(
    fun: Callable,
    x0: jnp.ndarray,
    *,
    method: str,
    tol: float = None,
    options: dict = None,
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


def _pre_fit_check(**kwargs):
    S = kwargs["S"]
    U = kwargs["U"]
    Z = kwargs["Z"]
    rotated_Z = kwargs["rotated_Z"]
    M = kwargs["M"]

    if S is None:
        raise ValueError("Missing singular values")
    if (U is None or Z is None) and (rotated_Z is None):
        raise ValueError("Argument missing for Z rotation")
    else:
        if M is None and (U is None or Z is None):
            raise ValueError("Unknown # markers")


def _rotate(U: np.ndarray, S: np.ndarray, Z: np.ndarray):

    D_sqrt = jnp.sqrt(jnp.sum(U ** 2 * S ** 2, axis=1))
    rotated_Z = U.T @ (D_sqrt * Z)

    return rotated_Z


# Model
def _negloglik(
    param: List[float],
    rotated_Z: np.ndarray,
    S: np.ndarray,
    M: int,
    constraints: dict,
) -> float:
    """Compute negative log likelihood

    Args:
        param: Values of parameters at which likelihood is evalulated
        constraints: constrainst on param

    """

    param = jnp.exp(jnp.array(param))
    param_ = {}
    param_.update({"h2a": param[0], "intercept": param[1]})
    param_.update(constraints)

    nongenetics = jnp.ones(rotated_Z.shape[0]) * param_["intercept"]
    scale = jnp.sqrt(param_["h2a"] / M * (S ** 2) + nongenetics)

    neglogp = -jax.scipy.stats.norm.logpdf(rotated_Z / S, loc=0.0, scale=scale)
    negloglik_ = jnp.sum(neglogp)

    return negloglik_


class HAMSTA:
    """Heritability estimation from admixture mapping summary statistics

    Args:
        k: Max number of singular values to be used
        S_thres: minimum value of S to be used

    """

    def __init__(self, k: int = None, S_thres: float = 1e-3, **kwargs):
        self.k = k
        self.S_thres = S_thres

    def summary(self):
        """Summarize estimation result"""
        pass

    # Estimation
    def fit(
        self,
        Z: np.ndarray = None,
        rotated_Z: np.ndarray = None,
        U: np.ndarray = None,
        S: np.ndarray = None,
        M: int = None,
        constraints: dict = {},
        jackknife=False,
    ):
        """Fit to compute likelihood and MLE

        Args:
            Z: signed test statistics of shape (M, )
            rotated_Z: test statistics after rotation of shape S.shape
            U: the matrix U from SVD results of A = USV'
            S: the matrix S from SVD results of A = USV'
            M: Number of markers


        """
        # check if rotated_Z, M, S are available or can be derived
        # =====
        _pre_fit_check(**locals())

        # prepare arguments for optimization
        # =======
        if rotated_Z is None:
            rotated_Z = _rotate(S=S, Z=Z, U=U)

        if U is not None:
            M = M or U.shape

        param0 = jnp.array([-0.7, -0.7])

        if M is None or rotated_Z is None or S is None:
            raise ValueError("Not enough arguments to start estimation")

        # Optimization
        # ============
        # H1 hypothesis
        # -------------
        obj_fun: Callable = partial(
            _negloglik, rotated_Z=rotated_Z, S=S, M=M, constraints=constraints
        )
        est_res = _minimize(obj_fun, x0=param0, method="trust-ncg")
        parameter = np.exp(est_res.x)
        h1 = -est_res.fun

        # H0 hypothesis
        # -------------
        constraints0 = constraints.copy()
        constraints0.update({"h2a": 0.0})
        obj_fun0: Callable = partial(
            _negloglik, rotated_Z=rotated_Z, S=S, M=M, constraints=constraints0
        )
        est_res = _minimize(obj_fun0, x0=param0, method="trust-ncg")
        h0 = -est_res.fun

        # store results
        # =====
        self.parameters = parameter
        self.h0 = h0
        self.h1 = h1
        self.p = self.lrt(h0, h1, len(constraints0) - len(constraints))["p"]

        # jackknife
        # =========
        if jackknife:
            self._jackknife(
                param_full=parameter,
                rotated_Z=rotated_Z,
                S=S,
                M=M,
                constraints=constraints,
            )

        return self

    def lrt(self, ell0, ell1, dof):
        """Likelihood ratio test"""

        chisq = -2 * (ell0 - ell1)
        p = 1 - stats.chi2.cdf(chisq, dof)

        return {"chisq": chisq, "dof": dof, "p": p}

    def _jackknife(
        self,
        param_full,
        rotated_Z: np.ndarray,
        S: np.ndarray,
        M: int,
        constraints: dict = {},
        num_blocks=10,
    ):

        pseudo_vals = []

        k = S.shape[0]

        for i in range(num_blocks):
            selected_index = np.repeat(True, k)
            selected_index[i::num_blocks] = False
            pseudo_hamsta = HAMSTA(**vars(self))

            pseudo_hamsta.fit(
                rotated_Z=rotated_Z[selected_index],
                M=M,
                S=S[selected_index],
                jackknife=False,
            )
            pseudo_val = (
                num_blocks * param_full - (num_blocks - 1) * pseudo_hamsta.parameters
            )
            pseudo_vals.append(pseudo_val)

        pseudo_vals = np.array(pseudo_vals)
        jk_se = np.sqrt(1 / num_blocks * np.var(pseudo_vals, ddof=1, axis=0))

        self.jk_se = jk_se

        return self
