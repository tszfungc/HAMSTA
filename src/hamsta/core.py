from functools import partial
from typing import Any, Callable, Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from jax import grad, jit, random
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


def rotate(U: np.ndarray, S: np.ndarray, Z: np.ndarray, residual_var: float, N: int):

    S = S * jnp.sqrt(N)
    D_sqrt = jnp.sqrt(jnp.sum(U ** 2 * S ** 2, axis=1))
    rotated_Z = np.sqrt(residual_var) * U.T @ (D_sqrt * Z)

    return rotated_Z


def _lrt(ell0, ell1, dof):
    """Likelihood ratio test"""

    chisq = -2 * (ell0 - ell1)
    p = 1 - stats.chi2.cdf(chisq, dof)

    return {"chisq": chisq, "dof": dof, "p": p}


# Model
def _negloglik(
    param: List[float],
    rotated_Z: np.ndarray,
    S: np.ndarray,
    M: int,
    constraints: dict,
    intercept_design: jnp.ndarray,
) -> float:
    """Compute negative log likelihood

    Args:
        param: Values of parameters at which likelihood is evalulated
        constraints: constrainst on param

    """

    param = jnp.exp(jnp.array(param))
    param_ = {}
    param_.update({"h2a": param[0], "intercept": param[1:]})
    param_.update(constraints)

    genetics = param_["h2a"] / M * (S ** 2)
    nongenetics = intercept_design @ param_["intercept"]
    total_var = genetics + nongenetics
    scale = jnp.sqrt(total_var)
    # nongenetics = jnp.linspace(0.5, 1.5, param_["intercept"].shape[0] + 2)[1:-1]
    # total_var = genetics + (nongenetics @ param_["intercept"])
    # nongenetics = jnp.ones(rotated_Z.shape[0]) * param_["intercept"]
    # scale = jnp.sqrt(param_["h2a"] / M * (S ** 2) + nongenetics)

    neglogp = -jax.scipy.stats.norm.logpdf(rotated_Z / S, loc=0.0, scale=scale)
    negloglik_ = jnp.sum(neglogp)

    return negloglik_


class HAMSTA:
    """Heritability estimation from admixture mapping summary statistics

    Args:
        k: Max number of singular values to be used
        S_thres: minimum value of S to be used

    """

    def __init__(
        self, k: int = None, S_thres: float = 1e-3, intercept_blksize=500, **kwargs
    ):
        self.k = k
        self.S_thres = S_thres
        self.result: Dict[str, Any] = {}
        self.intercept_blksize = intercept_blksize

    def to_dict(self):
        """Summarize estimation result in dict"""
        return self.result

    def to_dataframe(self):
        """Summarize estimation result in dataframe"""
        # all results are either single object or np.ndarray
        pd_dict = {}
        for k, v in self.result.items():
            if isinstance(v, np.ndarray):
                for idx, vv in enumerate(v):
                    pd_dict.update({f"{k}_{idx}": vv})
            else:
                pd_dict.update({k: v})

        result_df = pd.DataFrame(pd_dict, index=[0])

        return result_df

    # Estimation
    def fit(
        self,
        S: np.ndarray,
        intercept_design: jnp.ndarray,
        Z: np.ndarray = None,
        rotated_Z: np.ndarray = None,
        U: np.ndarray = None,
        N: int = None ,
        M: int = None,
        constraints: dict = {},
        residual_var: float = 1.0,
        jackknife: bool = False,
        est_thres: Union[bool, float] = False,
    ):
        """Fit to compute likelihood and MLE

        Args:
            Z: signed test statistics of shape (M, )
            rotated_Z: test statistics after rotation of shape S.shape
            U: the matrix U from SVD results of A = USV'
            S: the matrix S from SVD results of A = USV'
            M: Number of markers
            constraints: constraints applied in the optimization
            residual_var: variance of the residual in admixture mapping (default: 1)
            jackknife: If true, compute the jackknife standard error
            est_thres:
                If float, estimate significant threshold
                at family-wise error rate equal the float value.
                If true, assume FWER=0.05.
                If false, skip significant threshold estimation.



        """
        # check if rotated_Z, M, S are available or can be derived
        # =====
        _pre_fit_check(**locals())

        # prepare arguments for optimization
        # =======
        if rotated_Z is None:
            rotated_Z = rotate(S=S, Z=Z, U=U, residual_var=residual_var)

        S_filter = S > self.S_thres
        S = S[S_filter]
        S = S * jnp.sqrt(N)

        if U is not None:
            M = M or U.shape[0]
            U = U[:, S_filter]

        if M is None or rotated_Z is None or S is None:
            raise ValueError("Not enough arguments to start estimation")

        # apply filter
        rotated_Z = rotated_Z[S_filter]
        intercept_design = intercept_design[S_filter, :]

        # group intercept into multiple var components
        # bin_idx = np.arange(S.shape[0]) // self.intercept_blksize
        # group the last incomplete to the previous bin
        # if S.shape[0] % self.intercept_blksize != 0:
        #     bin_idx[bin_idx == max(bin_idx)] -= 1
        # intercept_design = pd.get_dummies(bin_idx).values
        # intercept_design = jnp.array(intercept_design)

        # Optimization
        # ============
        # decide multi intercept vs single intercept first, then test h2
        # Initial values
        # --------------
        param0 = jnp.repeat(-0.7, 1 + intercept_design.shape[1])
        # H1 multi intercept hypothesis
        # -------------
        obj_fun: Callable = partial(
            _negloglik,
            rotated_Z=rotated_Z,
            S=S,
            M=M,
            constraints=constraints,
            intercept_design=intercept_design,
        )
        est_res = _minimize(obj_fun, x0=param0, method="trust-ncg")
        h1_mintcpt = -est_res.fun

        # H0_intercept hypothesis
        # -------------
        intercept_design_null = jnp.ones((S.shape[0], 1))
        param0 = jnp.repeat(-0.7, 2)
        # constraints_intercept = constraints.copy()
        # constraints_intercept.update({"intercept": jnp.array([1.0])})
        obj_fun0_intercept: Callable = partial(
            _negloglik,
            rotated_Z=rotated_Z,
            S=S,
            M=M,
            constraints={},
            intercept_design=intercept_design_null,
        )
        est_res0 = _minimize(obj_fun0_intercept, x0=param0, method="trust-ncg")
        h1_sintcpt = -est_res0.fun

        # test multiple intercept vs single intercept to decide which one to proceed
        p_intercept = _lrt(h1_sintcpt, h1_mintcpt, intercept_design.shape[1] - 1)["p"]
        if p_intercept > 0.05:
            intercept_design = intercept_design_null
            parameter = np.exp(est_res0.x)
            h1 = h1_sintcpt
        else:
            parameter = np.exp(est_res.x)
            h1 = h1_mintcpt

        mean_intercept = jnp.mean(parameter[1:])
        intercepts = intercept_design @ parameter[1:]

        # H0_h2a hypothesis
        # -------------
        constraints0 = constraints.copy()
        constraints0.update({"h2a": 0.0})
        obj_fun0: Callable = partial(
            _negloglik,
            rotated_Z=rotated_Z,
            S=S,
            M=M,
            constraints=constraints0,
            intercept_design=intercept_design,
        )
        est_res = _minimize(obj_fun0, x0=param0, method="trust-ncg")
        h0 = -est_res.fun
        p_h2a = _lrt(h0, h1, len(constraints0) - len(constraints))["p"]

        # store results
        # =====
        self.result.update(
            {
                "parameter": parameter,
                "SE": [None, None],
                "mean_intercept": mean_intercept,
                "h0": h0,
                "h1_sintcpt": h1_sintcpt,
                "h1_mintcpt": h1_mintcpt,
                "p_h2a": p_h2a,
                "p_intercept": p_intercept,
            }
        )

        # jackknife
        # =========
        if jackknife:
            se = self._jackknife(
                param_full=parameter,
                rotated_Z=rotated_Z,
                S=S,
                intercept_design=intercept_design,
                M=M,
                constraints=constraints,
            )
            self.result.update({"SE": se})

        # Estimate sign threshold
        # =======================
        thres = np.nan
        if isinstance(est_thres, bool) and U is not None:
            if est_thres is True:
                thres = self.compute_thres(
                    fwer=0.05, U=U, S=S, intercept=intercepts, resid_var=residual_var
                )
        elif isinstance(est_thres, float) and U is not None:
            thres = self.compute_thres(
                fwer=0.05, U=U, S=S, intercept=intercepts, resid_var=residual_var
            )

        self.result.update({"thres": thres})

        return self

    def _jackknife(
        self,
        param_full,
        rotated_Z: np.ndarray,
        S: np.ndarray,
        intercept_design: np.ndarray,
        M: int,
        constraints: dict = {},
        num_blocks=10,
    ):

        pseudo_vals = []
        hyperparam = vars(self).copy()
        if "result" in hyperparam:
            hyperparam.pop("result")

        k = S.shape[0]

        for i in range(num_blocks):
            selected_index = np.repeat(True, k)
            selected_index[i::num_blocks] = False
            pseudo_hamsta = HAMSTA(**hyperparam)

            pseudo_hamsta.fit(
                rotated_Z=rotated_Z[selected_index],
                M=M,
                S=S[selected_index],
                jackknife=False,
                intercept_design=intercept_design[selected_index],
            )
            pseudo_val = (
                num_blocks * param_full
                - (num_blocks - 1) * pseudo_hamsta.result["parameter"]
            )
            pseudo_vals.append(pseudo_val)

        pseudo_vals = np.array(pseudo_vals)
        jk_se = np.sqrt(1 / num_blocks * np.var(pseudo_vals, ddof=1, axis=0))

        return jk_se

    def compute_thres(
        self,
        fwer: float,
        U: jnp.ndarray,
        S: jnp.ndarray,
        intercept: jnp.ndarray,
        resid_var: float = 1.0,
        rep: int = 2000,
    ) -> float:
        """Compute significant threshold at given family-wise error rate

        Args:
            fwer: family-wise error rate.

        Returns:
            Significance threshold

        """

        main_key = random.PRNGKey(123)
        main_key, sub_key = random.split(main_key)

        normal_seed = random.normal(key=sub_key, shape=(rep, S.shape[0])) * jnp.sqrt(
            intercept
        )
        Drt = 1 / jnp.sqrt(jnp.sum((U * S) ** 2, axis=1))
        simu_Z = (
            Drt * jnp.einsum("ij,j,kj -> ki", U, S, normal_seed) / jnp.sqrt(resid_var)
        )

        thres = jnp.percentile(jnp.max(simu_Z ** 2, axis=1), 100 * (1 - fwer))

        thres_p = 1 - stats.chi2.cdf(thres, df=1)

        return thres_p
