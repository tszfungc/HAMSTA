import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hamsta import io

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


if __name__ == "__main__":
    # CLI for debug only
    U = np.load(sys.argv[1] + ".SVD.U.npy")
    SDpj = np.load(sys.argv[1] + ".SVD.SDpj.npy")
    Z = io.read_sumstat(sys.argv[2])["T_STAT"].values
    print(rotate_Z(U, SDpj, Z))
