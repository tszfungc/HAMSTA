"""Estimation of the paramters h2a and interecepts
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from tqdm import tqdm


def model(M, S, intercept_design, rotated_z=None):
    """Model function of the rotated z.

    Likelihood function of the rotated Z scores.
    This function does not return anythings.

    Args:
        M: Number of local ancestry markers.
        S: singular values of the local ancestry matrix with global ancestry projected out.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501
    h2A = numpyro.param("h2A", 0.5, constraint=dist.constraints.greater_than(0.0))

    intercepts = numpyro.param(
        "intercepts",
        jnp.repeat(0.5, intercept_design.shape[1]),
        constraint=dist.constraints.greater_than(0.0),
    )

    with numpyro.plate("latent", S.shape[0]):
        nongenetics = intercept_design @ intercepts

        obs = numpyro.sample(  # noqa: F841
            "obs",
            dist.Normal(0.0, jnp.sqrt(h2A / M * S ** 4 + nongenetics * S ** 2)),
            obs=rotated_z,
        )


def guide(M, S, intercept_design, rotated_z=None):
    pass


def estimate(M, S, rotated_z=None, binsize=500, jackknife=True):
    """Find maximum likelihood estimates

    Args:
        M: Number of local ancestry markers.
        S: path to a npy file readable by ``numpy.load`` that stores 1D numpy array of singular values of the local ancestry matrix with global ancestry projected out.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
    """  # noqa: E501

    S = jnp.array(S)
    select_S = S > 1e-3

    print(S.shape[0], sum(select_S))

    # filter out small number selected_S
    rotated_z = rotated_z[select_S]
    S = S[select_S]

    # Prepare design matrix for intercepts
    bin_idx = np.arange(rotated_z.shape[0]) // binsize
    # group the last incomplete to the previous bin
    if rotated_z.shape[0] % binsize != 0:
        bin_idx[bin_idx == max(bin_idx)] -= 1
    bin_idx_design_mat = pd.get_dummies(bin_idx).values

    # to be wrapped in function to allow jackknife
    # inference object
    adam = Adam(5e-3)
    svi = SVI(model=model, guide=guide, loss=Trace_ELBO(), optim=adam)

    # input file

    seed = np.random.randint(1000)
    main_key = jax.random.PRNGKey(seed)
    main_key, sub_key = jax.random.split(main_key)

    res = svi.run(
        sub_key,
        5000,
        M=M,
        S=S,
        intercept_design=bin_idx_design_mat,
        rotated_z=rotated_z,
        progress_bar=True,
    )

    # print(res.losses[1:]-res.losses[:-1] )
    n_steps_converged = jnp.where(res.losses[1:] - res.losses[:-1] > -1e-3)[0][0]

    h2a_jk = []
    for i in tqdm(range(S.shape[0])):
        main_key, sub_key = jax.random.split(main_key)
        h2a_jk.append(
            svi.run(
                sub_key,
                n_steps_converged,
                M=M,
                S=np.delete(S, i, 0),
                intercept_design=np.delete(bin_idx_design_mat, i, 0),
                rotated_z=np.delete(rotated_z, i, 0),
                progress_bar=False,
            ).params["h2A"]
        )
    h2a_jk = np.array(h2a_jk)
    n = S.shape[0]
    h2a_se = np.sqrt((n - 1) / n * np.sum((h2a_jk - np.mean(h2a_jk)) ** 2))

    print(f"PRNGKey seed: {seed}")
    print(f"h2a estimate: {res.params['h2A']} (se: {h2a_se})")
    print(f"intercepts estimate: {res.params['intercepts']}", flush=True)
    print(f"mean intercepts estimate: {np.mean(res.params['intercepts'])}", flush=True)

    return res
