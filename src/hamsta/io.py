import logging
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

_logger = logging.getLogger(__name__)


def read_bed(fname: str):
    bedf = open(fname, 'r')
    return [list(map(int, line.split("\t"))) for line in bedf.read().splitlines()]

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
    Z_colname: str = "Z",
    **pd_kwargs,
) -> jnp.ndarray:

    """Summary statistics reader

    Args:
        fname: filename of the genome-wide scan results
        Z_colname: colname of the Z
        pd_kwargs: keyword arguments passed to pandas

    """

    default_pd_kwargs = {"sep": "\t"}
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


def read_global_ancestry(fname: str, sample_colname: str, **pd_kwargs) -> pd.DataFrame:
    """Global ancestry reader

    | The sample column will be stored in strings.
    | All other keyword arguments will be passed to pandas.read_csv

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

    default_pd_kwargs = {"sep": "\t"}
    default_pd_kwargs.update(pd_kwargs)
    Q_df = pd.read_csv(fname, **default_pd_kwargs, dtype={sample_colname: str})
    Q_df = Q_df.rename({sample_colname: "sample"}, axis=1)

    Q_df = Q_df.set_index("sample").reset_index()

    return Q_df


def read_rfmixfb(
    fname: str,
    ancestry: str,
    exclude: str = None,
) -> Tuple[jnp.ndarray, pd.DataFrame]:
    """Reader for RFMIX .fb.tsv output

    Reading RFMIX .fb.tsv output as probability dosages

    Args:
        fname: Path to RFMIX output
        ancestry: The ancestry to be extracted

    Returns:
        a local ancestry matrix (marker, sample) and list of sample


    Example
    -------
    >>> from hamsta import io
    >>> A, A_sample = io.read_rfmixfb("tests/testdata/example.fb.tsv", "HCB")
    >>> A[:5, :5]
    DeviceArray([[2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ]], dtype=float32)
    >>> A_sample.head(5)
       sample
    0  HCB182
    1  HCB190
    2  HCB191
    3  HCB193
    4  HCB194


    """

    # Read ancestry line
    f_handle = open(fname, "r")
    comment = f_handle.readline()
    pops = comment.strip().split("\t")[1:]
    n_pops = len(pops)
    popdict = {pops[i]: i for i in range(n_pops)}
    popidx = popdict[ancestry]

    # header line
    # RFMIX output dim: (marker , (sample x ploidy x ancestry))
    header = f_handle.readline()
    indiv = list(map(lambda x: x.split(":::")[0], header.strip().split("\t")[4:]))
    indiv = np.array(indiv[:: (2 * n_pops)], dtype=str)
    N = indiv.shape[0]

    # data lines
    # Reshape to (marker, sample, ploidy, ancestry) array
    # choose ancestry and sum over ploidy
    LA_matrix = []  # read into (marker by (sample x ploidy x ancestry))
    for i, line in enumerate(f_handle):
        if i % 100 == 0:
            logging.debug(f"processing {i}-th marker")

        line_split = line.strip().split("\t")
        LA_matrix.append(line_split[4:])

    LA_matrix = jnp.float_(LA_matrix).reshape(-1, N, 2, n_pops)
    LA_matrix = LA_matrix[:, :, :, popidx]
    LA_matrix = jnp.sum(LA_matrix, axis=2)

    # make individual list
    sample_df = pd.DataFrame({"sample": indiv})

    return LA_matrix, sample_df


def read_zarr(
    fname: str,
    ancestry: str,
    exclude: str = None,
) -> Tuple[jnp.ndarray, pd.DataFrame]:
    """Reader for xarray stored in .zarr

    Read a :class:`xarray.Dataset` with data ``locanc`` in
    (marker, sample, ploidy, ancestry), example::

        <xarray.Dataset>
        Dimensions:   (marker: 8, sample: 39, ploidy: 2, ancestry: 2)
        Coordinates:
        * ancestry  (ancestry) <U3 'HCB' 'JPT'
        * marker    (marker) uint32 1 6 12 20 25 31 36 43
        * ploidy    (ploidy) int8 0 1
        * sample    (sample) <U6 'HCB182' 'HCB190' 'HCB191' ... 'JPT266' 'JPT267'
        Data variables:
            locanc    (marker, sample, ploidy, ancestry) float32 1.0 0.0 1.0 ... 0.0 1.0

    Args:
        fname: Path to RFMIX output
        ancestry: The ancestry to be extracted

    Returns:
        a local ancestry matrix (marker, sample) and list of sample

    Example
    -------
    >>> from hamsta import io
    >>> A, A_sample = io.read_zarr("tests/testdata/example.zarr", "HCB")
    >>> A[:5, :5]
    DeviceArray([[2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ],
                 [2.   , 2.   , 2.   , 1.969, 1.   ]], dtype=float32)
    >>> A_sample.head(5)
       sample
    0  HCB182
    1  HCB190
    2  HCB191
    3  HCB193
    4  HCB194


    """
    ds = xr.open_zarr(fname)

    LA_matrix = jnp.array(ds.locanc.sel(ancestry=ancestry).sum(dim="ploidy"))

    sample_df = ds.sample.to_dataframe().reset_index(drop=True)

    return LA_matrix, sample_df


def read_nc(
    fname: str,
    ancestry: str,
    exclude: str=None,
) -> Tuple[jnp.ndarray, pd.DataFrame]:

    ds = xr.open_dataset(fname).load()

    #LA_matrix = jnp.array(ds.locanc.sel(ancestry=ancestry).sum(dim="ploidy"))
    ds_LA = ds.locanc.sel(ancestry=ancestry).sum(dim="ploidy")
    if exclude is not None:
        exclude_region = read_bed(exclude)
        extract = np.logical_and.reduce(
            [~np.logical_and(start <= ds_LA['marker'], ds_LA['marker'] < end)
             for _, start, end in exclude_region]
        )
        ds_LA = ds_LA[extract, :]

    LA_matrix = jnp.array(ds_LA)

    sample_df = ds.sample.to_dataframe().reset_index(drop=True)

    return LA_matrix, sample_df


if __name__ == "__main__":
    pass
