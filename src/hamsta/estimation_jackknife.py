"""Estimation of the paramters h2a and interecepts
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import config
from scipy import stats
from tqdm import tqdm

from hamsta import utils

config.update("jax_enable_x64", True)


def neg_log_lik(x, M, S, intercept_design, rotated_z):
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
    intercepts = x[1:]

    nongenetics = intercept_design @ intercepts

    scale = jnp.sqrt(h2a / M * S ** 4 + nongenetics * S ** 2)

    neglogp = -jax.scipy.stats.norm.logpdf(rotated_z, loc=0, scale=scale)
    negloglik_ = jnp.sum(neglogp)

    return negloglik_


def estimate(M, S, design_matrix, rotated_z=None):
    """Find maximum likelihood estimates

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        intercept_design: A design matrix indicating the which bin each rotated z belongs to.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    # input file
    x0 = jnp.repeat(-0.7, design_matrix.shape[1] + 1)

    obj_fun = partial(
        neg_log_lik, M=M, S=S, intercept_design=design_matrix, rotated_z=rotated_z
    )

    est_res = utils.minimize(obj_fun, x0=x0, method="trust-ncg")
    est_res_x = jnp.exp(jnp.array(est_res.x))

    return est_res_x


def jackknife(M, S, design_matrix, rotated_z=None):
    """Block Jackknife returning point estimates and s.e.

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        intercept_design: A design matrix indicating the which bin each rotated z belongs to.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    # point estimation
    est = estimate(M, S, design_matrix, rotated_z)

    # pseudovalues
    n_block = 20
    k = S.shape[0]

    pseudo_val = []
    for i in tqdm(range(n_block)):
        selected_index = np.repeat(True, k)
        selected_index[i::n_block] = False
        est_delblock = estimate(
            M,
            S[selected_index],
            design_matrix[selected_index],
            rotated_z[selected_index],
        )
        pseudo_val.append(n_block * est[0] - (n_block - 1) * est_delblock[0])

    # Block Jackknife standard error
    h2a_se = np.sqrt(1 / n_block * np.var(pseudo_val, ddof=1))

    return est, h2a_se


def run(M, S, rotated_z=None, binsize=500):
    S = jnp.array(S)
    select_S = S > 1e-3

    # filter out small number selected_S
    rotated_z = rotated_z[select_S]
    S = S[select_S]

    # Prepare design matrix for intercepts
    bin_idx = np.arange(rotated_z.shape[0]) // binsize
    # group the last incomplete to the previous bin
    if rotated_z.shape[0] % binsize != 0:
        bin_idx[bin_idx == max(bin_idx)] -= 1
    bin_idx_design_mat = pd.get_dummies(bin_idx).values

    S = jnp.array(S)
    bin_idx_design_mat = jnp.array(bin_idx_design_mat)
    rotated_z = jnp.array(rotated_z)

    # Estimation
    est_res_x, h2a_se = jackknife(
        M=M, S=S, design_matrix=bin_idx_design_mat, rotated_z=rotated_z
    )

    # LRT
    H1_negloglik = neg_log_lik(
        M=M,
        S=S,
        intercept_design=bin_idx_design_mat,
        rotated_z=rotated_z,
        x=jnp.log(est_res_x),
    )

    H0_negloglik = neg_log_lik(
        M=M,
        S=S,
        intercept_design=bin_idx_design_mat,
        rotated_z=rotated_z,
        x=jnp.concatenate(
            [jnp.log(est_res_x[0, None]), jnp.repeat(0.0, est_res_x.shape[0] - 1)]
        ),
    )

    chi2 = 2 * (H0_negloglik - H1_negloglik)

    p = 1 - stats.chi2.cdf(chi2, df=est_res_x.shape[0] - 1)
    print(f"\nh2a estimate: {est_res_x[0]} (se: {h2a_se})")
    print(f"intercepts estimate: {est_res_x[1:]}")
    print(f"mean intercepts estimate: {jnp.mean(est_res_x[1:])}", flush=True)
    print(f"LRT chi2 = {chi2:.2f} and p = {p:.4e}")
