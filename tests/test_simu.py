import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from hamsta import HAMSTA, io, preprocess, simu


def test_simu():
    LAmatrix, sample_df = io.read_zarr(
        "../example_data/example.subset.concat.zarr/", "AFR"
    )

    Q = LAmatrix.mean(axis=0)[:, None] / 2

    U, S = preprocess.SVD(A=LAmatrix, Q=Q)

    pheno = simu.simu_pheno(A=LAmatrix, Q=Q, pve=jnp.array([5e-3, 0, 0]), rep=11)

    tstat = simu.assoc(y=pheno, x=LAmatrix.T, c=Q)

    ham = HAMSTA()

    simu_results = []
    for i in tqdm(range(10)):
        ham.fit(Z=tstat[:, i], S=S, U=U, jackknife=False)
        simu_results.append(ham.to_dataframe())

    simu_results = pd.concat(simu_results)
    print(simu_results)
