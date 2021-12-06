"""Estimation of the paramters h2a and interecepts
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy import optimize
from tqdm import tqdm


def neg_log_lik(log_x, M, S, intercept_design, rotated_z):
    """Likelihood function of the rotated z.

    Args:
        M: Number of local ancestry markers.
        S: singular values of the local ancestry matrix with global ancestry projected out.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    return:
        negloglik_: negative log likelihood
    """  # noqa: E501

    x = jnp.exp(log_x)

    h2a = x[0]
    intercepts = x[1:]

    nongenetics = intercept_design @ intercepts

    scale = jnp.sqrt(h2a / M * S ** 4 + nongenetics * S ** 2)

    neglogp = -jax.scipy.stats.norm.logpdf(rotated_z, loc=0, scale=scale)
    negloglik_ = jnp.sum(neglogp)

    return negloglik_


def run(M, S, rotated_z=None, binsize=500):
    S = jnp.array(S)
    select_S = S > 1e-3

    # print(S.shape[0], sum(select_S))

    # filter out small number selected_S
    rotated_z = rotated_z[select_S]
    S = S[select_S]

    # Prepare design matrix for intercepts
    bin_idx = np.arange(rotated_z.shape[0]) // binsize
    # group the last incomplete to the previous bin
    if rotated_z.shape[0] % binsize != 0:
        bin_idx[bin_idx == max(bin_idx)] -= 1
    bin_idx_design_mat = pd.get_dummies(bin_idx).values

    est_res_x = estimate(
        M=M, S=S, design_matrix=bin_idx_design_mat, rotated_z=rotated_z
    )

    h2a_se = bootstrap(M=M, S=S, design_matrix=bin_idx_design_mat, rotated_z=rotated_z)

    print(f"h2a estimate: {est_res_x[0]} (se: {h2a_se})")
    print(f"intercepts estimate: {est_res_x[1:]}")
    print(f"mean intercepts estimate: {jnp.mean(est_res_x[1:])}", flush=True)


def estimate(M, S, design_matrix, rotated_z=None):
    """Find maximum likelihood estimates

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    # input file
    x0 = jnp.repeat(jnp.log(0.5), design_matrix.shape[1] + 1)
    est_res = optimize.minimize(
        neg_log_lik, x0=x0, args=(M, S, design_matrix, rotated_z), method="BFGS"
    )
    est_res_x = jnp.exp(jnp.array(est_res.x))

    return est_res_x


def bootstrap(M, S, design_matrix, rotated_z=None):
    x0 = jnp.repeat(jnp.log(0.5), design_matrix.shape[1] + 1)

    h2a_bs = []
    for i in tqdm(range(5)):
        resampled = np.random.choice(np.arange(S.shape[0]), S.shape[0], replace=True)
        bs_res = optimize.minimize(
            neg_log_lik,
            x0=x0,
            args=(M, S[resampled], design_matrix[resampled], rotated_z[resampled]),
            method="BFGS",
        )
        h2a_bs.append(bs_res.x[0])

    h2a_se = np.std(np.exp(h2a_bs))

    return h2a_se
