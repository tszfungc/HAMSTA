"""Estimation of the paramters h2a and interecepts
"""
# TO DO
# organize arguments passing to estimate()
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from scipy import stats
from tqdm import tqdm

from hamsta import utils

config.update("jax_enable_x64", True)


def neg_log_lik(x, N, M, S, intercept_design, rotated_z, fix_intercept=False):
    """Negative Log Likelihood function of the rotated z.

    Args:
        M: Number of local ancestry markers.
        S: Singular values of the local ancestry matrix with global ancestry projected out.
        intercept_design: A design matrix indicating the which bin each rotated z belongs to.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
        x: parameter in log scale.
    return:
        negloglik_: negative log likelihood
    """  # noqa: E501

    x = jnp.exp(x)

    h2a = x[0]

    if fix_intercept:
        nongenetics = intercept_design.ravel()
    else:
        intercepts = x[1:]
        nongenetics = intercept_design @ intercepts

    scale = jnp.sqrt(h2a / M * (S ** 2) + nongenetics)

    neglogp = -jax.scipy.stats.norm.logpdf(rotated_z / S, loc=0.0, scale=scale)
    negloglik_ = jnp.sum(neglogp)

    return negloglik_


def neg_log_lik_H0(x, **args):
    return neg_log_lik(jnp.concatenate([jnp.array([-jnp.inf]), x]), **args)


def estimate(N, M, S, design_matrix, rotated_z=None, fix_intercept=False, hypothesis=1):
    """Find maximum likelihood estimates

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        intercept_design: A design matrix indicating the which bin each rotated z belongs to.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    lik_function_options = [neg_log_lik_H0, neg_log_lik]

    x0_options = [
        jnp.array([-0.7]),
        jnp.concatenate((jnp.array([-0.7]), jnp.repeat(-0.7, design_matrix.shape[1]))),
    ]

    obj_fun = partial(
        lik_function_options[hypothesis],
        N=N,
        M=M,
        S=S,
        intercept_design=design_matrix,
        rotated_z=rotated_z,
        fix_intercept=fix_intercept,
    )

    x0 = x0_options[hypothesis]

    # if fix_intercept:
    #     x0 = jnp.array([-0.7])
    # else:
    #     x0 = jnp.concatenate((
    #         jnp.array([-0.7]),
    #         jnp.repeat(-0.7, design_matrix.shape[1])
    #     ))

    est_res = utils.minimize(obj_fun, x0=x0, method="trust-ncg")
    est_res_x = jnp.exp(jnp.array(est_res.x))

    return est_res_x, est_res


def jackknife(N, M, S, design_matrix, rotated_z=None, n_block=10, fix_intercept=False):
    """Block Jackknife returning point estimates and s.e.

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        intercept_design: A design matrix indicating the which bin each rotated z belongs to.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    # point estimation
    est, optres = estimate(
        N, M, S, design_matrix, rotated_z, fix_intercept=fix_intercept
    )
    print(f"Optimization success: {optres.success}")
    print(f"{optres.message}")

    k = S.shape[0]

    pseudo_val = []
    pseudo_val_int = []
    for i in tqdm(range(n_block)):
        selected_index = np.repeat(True, k)
        selected_index[i::n_block] = False
        est_delblock, _ = estimate(
            N,
            M,
            S[selected_index],
            design_matrix[selected_index],
            rotated_z[selected_index],
            fix_intercept=fix_intercept,
        )
        pseudo_val.append(n_block * est[0] - (n_block - 1) * est_delblock[0])
        pseudo_val_int.append(
            n_block * np.mean(est[1:]) - (n_block - 1) * np.mean(est_delblock[1:])
        )

    # Block Jackknife standard error
    h2a_se = np.sqrt(1 / n_block * np.var(pseudo_val, ddof=1))
    s2e_se = np.sqrt(1 / n_block * np.var(pseudo_val_int, ddof=1))

    return optres, h2a_se, s2e_se


def LRT(opt_res1, opt_res0):
    H1, H0 = opt_res1.fun, opt_res0.fun
    chi2 = 2 * (H0 - H1)
    df = len(opt_res1.x) - len(opt_res0.x)
    p = 1 - stats.chi2.cdf(chi2, df=df)
    return {"chi2": chi2, "p": p}


def run(N, M, S, rotated_z=None, binsize=500, yvar=1, fix_intercept=False):
    S = jnp.array(S)
    select_S = S > 1.0

    # filter out small number selected_S
    rotated_z = rotated_z[select_S]
    rotated_z = jnp.array(rotated_z)

    S = S[select_S]
    S = jnp.array(S)

    # H0: h2a=0, single intercept
    H0_est, H0_est_res = estimate(
        N=N,
        M=M,
        S=S,
        design_matrix=jnp.ones((S.shape[0], 1)),
        rotated_z=rotated_z,
        hypothesis=0,
    )

    # H1: h2a, single intercept
    H1_est_res, H1_h2a_se, H1_s2e_se = jackknife(
        N=N, M=M, S=S, design_matrix=jnp.ones((S.shape[0], 1)), rotated_z=rotated_z
    )

    print(f"h2a: {H1_est_res.x[0]/yvar} (se: {H1_h2a_se/yvar})", flush=True)
    print(
        f"intercept: {jnp.mean(H1_est_res.x[1:])/yvar} (se: {H1_s2e_se/yvar})",
        flush=True,
    )

    print(f"Optimization success: {H1_est_res.success}")
    print(f"{H1_est_res.message}")

    # H2: h2a, multiple intercepts

    # Prepare design matrix for intercepts
    multi_intercept_design = utils.make_intercept_design(rotated_z.shape[0], binsize)

    # Estimation
    H2_est_res, H2_h2a_se, H2_s2e_se = jackknife(
        N=N,
        M=M,
        S=S,
        design_matrix=multi_intercept_design,
        rotated_z=rotated_z,
        fix_intercept=fix_intercept,
    )

    # LRT
    print("LRT H0: h2a = 0; single intercept")
    print("LRT H1: h2a !=0; single interecpt")
    print("LRT H2: h2a !=0; multiple interecpt")

    lrt_res1 = LRT(H1_est_res, H0_est_res)
    lrt_res2 = LRT(H2_est_res, H1_est_res)

    print(f"LRT H1/H0: chi2 = {lrt_res1['chi2']:.2f} and p = {lrt_res1['p']:.4e}")
    H1_x = np.exp(H1_est_res.x)
    print(
        "#RE1 "
        + f"{H1_x[0]/yvar} "
        + f"{H1_h2a_se/yvar} "
        + f"{np.mean(H1_x[1:])/yvar} "
        + f"{H1_s2e_se/yvar} "
        + f"{lrt_res1['p']:.4e}"
    )
    print(f"LRT H2/H1: chi2 = {lrt_res2['chi2']:.2f} and p = {lrt_res2['p']:.4e}")

    H2_x = np.exp(H2_est_res.x)
    print(
        "#RE2 "
        + f"{H2_x[0]/yvar} "
        + f"{H2_h2a_se/yvar} "
        + f"{np.mean(H2_x[1:])/yvar} "
        + f"{H2_s2e_se/yvar} "
        + f"{lrt_res2['p']:.4e}"
    )
