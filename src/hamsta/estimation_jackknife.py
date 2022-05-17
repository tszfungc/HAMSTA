"""Estimation of the paramters h2a and interecepts
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import config
from tqdm import tqdm

from hamsta import utils

config.update("jax_enable_x64", True)


def neg_log_lik(x,
                N,
                M,
                S,
                intercept_design,
                rotated_z,
                fix_intercept=False):
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

    neglogp = -jax.scipy.stats.norm.logpdf(rotated_z / S , loc=0. , scale=scale)
    negloglik_ = jnp.sum(neglogp)

    return negloglik_


def estimate(N, M, S, design_matrix, rotated_z=None, fix_intercept=False):
    """Find maximum likelihood estimates

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        intercept_design: A design matrix indicating the which bin each rotated z belongs to.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    obj_fun = partial(
            neg_log_lik,
            N=N,
            M=M,
            S=S,
            intercept_design=design_matrix,
            rotated_z=rotated_z,
            fix_intercept=fix_intercept
        )
    if fix_intercept:
        x0 = jnp.array([-0.7])
    else:
        x0 = jnp.concatenate((
            jnp.array([-0.7]),
            jnp.repeat(0., design_matrix.shape[1])
        ))

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
    est, optres = estimate(N,
                           M,
                           S,
                           design_matrix,
                           rotated_z,
                           fix_intercept=fix_intercept)
    print(f'Optimization success: {optres.success}')
    print(f'{optres.message}')

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
            fix_intercept=fix_intercept
        )
        pseudo_val.append(n_block * est[0] - (n_block - 1) * est_delblock[0])
        pseudo_val_int.append(
            n_block * np.mean(est[1:]) - (n_block - 1) * np.mean(est_delblock[1:])
        )

    # Block Jackknife standard error
    h2a_se = np.sqrt(1 / n_block * np.var(pseudo_val, ddof=1))
    s2e_se = np.sqrt(1 / n_block * np.var(pseudo_val_int, ddof=1))

    return est, h2a_se, s2e_se


def run(N, M, S, rotated_z=None, binsize=500, yvar=1, fix_intercept=False):
    S = jnp.array(S)
    select_S = S > 1.

    # filter out small number selected_S
    rotated_z = rotated_z[select_S]
    rotated_z = jnp.array(rotated_z)

    S = S[select_S]
    S = jnp.array(S)

    # H1: single intercept
    H1_est, H1_est_res = estimate(
        N=N,
        M=M,
        S=S,
        design_matrix=jnp.ones((S.shape[0], 1)),
        rotated_z=rotated_z,
        fix_intercept=fix_intercept)

    print(f'Optimization success: {H1_est_res.success}')
    print(f'{H1_est_res.message}')
    # H2

    # Prepare design matrix for intercepts
    bin_idx = np.arange(rotated_z.shape[0]) // binsize
    # group the last incomplete to the previous bin
    if rotated_z.shape[0] % binsize != 0:
        bin_idx[bin_idx == max(bin_idx)] -= 1
    bin_idx_design_mat = pd.get_dummies(bin_idx).values

    bin_idx_design_mat = jnp.array(bin_idx_design_mat)

    # Estimation
    H2_est, H2_h2a_se, H2_s2e_se = jackknife(
        N=N,
        M=M,
        S=S,
        design_matrix=bin_idx_design_mat,
        rotated_z=rotated_z,
        fix_intercept=fix_intercept
    )

    # LRT

    # H1_negloglik = neg_log_lik(
    #    N=N,
    #    M=M,
    #    S=S,
    #    intercept_design=bin_idx_design_mat,
    #    rotated_z=rotated_z,
    #    x=jnp.log(H1_est)
    # )

    # H2_negloglik = neg_log_lik(
    #    N=N,
    #    M=M,
    #    S=S,
    #    intercept_design=bin_idx_design_mat,
    #    rotated_z=rotated_z,
    #    x=jnp.log(H2_est)
    # )

    # DEBUG
#    print("Genetics", N/M*S**2*H2_est[0])
#    print("Non Genetics", H2_est[1])
#    print("scaled rZ", rotated_z/S)
#    print("scaled rZ var", np.var((rotated_z/S/np.sqrt(N))[:100]))

    print(f"\nh2a estimate: {H2_est[0]} (se: {H2_h2a_se} )")
    print(f"intercepts estimate: {H2_est[1:]} (se: {H2_s2e_se} )")
    print(f"mean intercepts estimate: {jnp.mean(H2_est[1:])}", flush=True)

    print(f"s2a/vP: {H2_est[0]/np.sum(H2_est)}", flush=True)
    print(f"s2a/vY: {H2_est[0]/yvar} (se: {H2_h2a_se/yvar})", flush=True)
    print(
        f"intercept/vY: {jnp.mean(H2_est[1:])/yvar} (se: {H2_s2e_se/yvar})",
        flush=True)

    print("--------")
    print('h2a if single intercept')
    print(f'Under H0: h2a : {H1_est[0]}')
    print("--------")

    # print('LRT H0: single intercept = 1')
    # chi2 = 2 * (H0_negloglik - H1_negloglik)
    # p = 1 - stats.chi2.cdf(chi2, df=1)
    # print(f"LRT1 chi2 = {chi2:.2f} and p = {p:.4e}")

    # print('LRT H0: equal intercept ')
    # chi2 = 2 * (H1_negloglik - H2_negloglik)
    # p = 1 - stats.chi2.cdf(chi2, df=bin_idx_design_mat.shape[1]-1)

