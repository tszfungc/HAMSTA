import numpy as np
import pandas as pd
import logging
import jax.numpy as jnp

_logger = logging.getLogger(__name__)


def read_singular_val(svdprefix, svdprefix_chr, nS):
    if svdprefix is not None:
        S_ = np.load(f"{svdprefix}.SVD.S.npy")[:nS]
    elif svdprefix_chr is not None:
        S_ = np.concatenate(
            [np.load(f"{svdprefix_chr}.{i}.SVD.S.npy")[:nS] for i in range(1, 23)]
        )
    else:
        S_ = None

    return S_


def read_sumstat(
    fname: str,
    Z_colname: str = 'Z', 
    **pd_kwargs,
    ) -> jnp.ndarray:

    """Summary statistics reader

    Args:
        fname: filename of the genome-wide scan results
        Z_colname: colname of the Z
        pd_kwargs: keyword arguments passed to pandas
    
    """

    default_pd_kwargs = {'sep': '\t'}
    default_pd_kwargs.update(pd_kwargs)

    df = pd.read_csv(fname, **default_pd_kwargs)

    Z_series = df[[Z_colname]]

    Z = Z_series.values.squeeze()
    Z[np.isnan(Z)] = 0.0
    Z = jnp.array(Z)

    return Z


def read_SVD_chr(svdprefix_chr, chrom=None):
    U = np.load(f"{svdprefix_chr}.{chrom}.SVD.U.npy")
    S = np.load(f"{svdprefix_chr}.{chrom}.SVD.S.npy")
    SDpj = np.load(f"{svdprefix_chr}.{chrom}.SVD.SDpj.npy")

    return U, S, SDpj


def read_SVD(svdprefix):
    U = np.load(svdprefix + ".SVD.U.npy")
    S = np.load(svdprefix + ".SVD.S.npy")
    SDpj = np.load(svdprefix + ".SVD.SDpj.npy")

    return U, S, SDpj


def read_global_ancestry(
    fname: str,
    sample_colname: str,
    **pd_kwargs
    ) -> pd.DataFrame:
    """Global ancestry reader

    The sample column will be stored in strings. All other keyword arguments will be passed to pandas.read_csv
    
    Args:
        fname: filename of the global ancestry file
        sample_colname: column names of the sample

    Example
    -------
    >>> from hamsta.io import read_global_ancestry
    >>> Q_df = read_global_ancestry("tests/testdata/example.rfmix.Q",
    ...     sample_colname="#sample", skiprows=1)
    >>> Q_df.head()
       sample  HCB  JPT
    0  HCB182  1.0  0.0
    1  HCB190  1.0  0.0
    2  HCB191  1.0  0.0
    3  HCB193  1.0  0.0
    4  HCB194  0.5  0.5
    >>> Q_df.dtypes
    sample     object
    HCB       float64
    JPT       float64
    dtype: object

    """

    _logger.info('testing')

    default_pd_kwargs = {'sep': '\t'}
    default_pd_kwargs.update(pd_kwargs)
    Q_df =  pd.read_csv(fname, **default_pd_kwargs, dtype={sample_colname: str})
    Q_df = Q_df.rename({sample_colname: 'sample'}, axis=1)

    Q_df = Q_df.set_index('sample').reset_index()

    return Q_df


if __name__ == "__main__":
    pass
