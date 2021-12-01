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


def model(M, S, rotated_z=None, binsize=500):
    """Model function of the rotated z.

    Likelihood function of the rotated Z scores.
    This function does not return anythings.

    Args:
        M: Number of local ancestry markers.
        S: singular values of the local ancestry matrix with global ancestry projected out.
        rotated_z: 1D numpy array of the rotated z scores. This is computated by np.sqrt(1-R2) * (U.T * Drt @ Z), where R2 is the phenotypic variance explained by global ancestry, U is the U matrix in SVD of A.T = U * S @ V.T, Drt is a 1D array of S.D. of local ancestry after global ancestry is projected out, and Z is the unrotated Z.
        binsize: the number of rotated Z scores in each bin sharing an intercept. Default: 500
    """  # noqa: E501
    h2A = numpyro.param("h2A", 0.5, constraint=dist.constraints.greater_than(0.0))

    intercepts = numpyro.param(
        "intercepts",
        jnp.repeat(0.5, rotated_z.shape[0] // binsize),
        constraint=dist.constraints.greater_than(0.0),
    )

    bin_idx = np.arange(rotated_z.shape[0]) // binsize
    # group the last incomplete to the previous bin
    if rotated_z.shape[0] % binsize != 0:
        bin_idx[bin_idx == max(bin_idx)] -= 1
    bin_idx_design_mat = pd.get_dummies(bin_idx).values

    with numpyro.plate("latent", S.shape[0]):
        nongenetics = bin_idx_design_mat @ intercepts

        obs = numpyro.sample(  # noqa: F841
            "obs",
            dist.Normal(0.0, jnp.sqrt(h2A / M * S ** 4 + nongenetics * S ** 2)),
            obs=rotated_z,
        )


def guide(M, S, rotated_z=None, binsize=500):
    pass


def estimate(M, S, rotated_z=None):
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

    # inference object
    adam = Adam(5e-3)
    svi = SVI(model=model, guide=guide, loss=Trace_ELBO(), optim=adam)

    # input file

    seed = np.random.randint(1000)
    main_key = jax.random.PRNGKey(seed)
    main_key, sub_key = jax.random.split(main_key)
    res = svi.run(sub_key, 5000, M, S=S, rotated_z=rotated_z, progress_bar=True)
    print(f"PRNGKey seed: {seed}")
    print(f"h2a estimate: {res.params['h2A']}")
    print(f"intercepts estimate: {res.params['intercepts']}", flush=True)
    print(f"mean intercepts estimate: {np.mean(res.params['intercepts'])}", flush=True)

    return res
