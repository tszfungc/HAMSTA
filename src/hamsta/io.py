import numpy as np
import pandas as pd
import pgenlib as pg


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


def read_sumstat(fname):
    df = pd.read_csv(fname, sep="\t")

    Z_colname = ["Z", "T_STAT", "STAT", "ZSCORE", "Z-SCORE", "GC_ZSCORE"]

    Z_series = df.iloc[:, np.in1d(df.columns, Z_colname)]
    assert Z_series.shape[1] == 1, "More than 1 columns of test statistcs found."

    # Add warning NA

    Z_ = Z_series.values.squeeze()
    Z_[np.isnan(Z_)] = 0.0

    return Z_


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


def read_pgen(f_pgen):
    reader = pg.PgenReader(f_pgen.encode("utf-8"))
    dosage_arr = np.empty(
        (reader.get_variant_ct(), reader.get_raw_sample_ct()), dtype=np.float
    )

    print(f"Reading pgen file (n, p): ({dosage_arr.shape[1]}, {dosage_arr.shape[0]})")

    for i in range(dosage_arr.shape[0]):
        reader.read_dosages(i, dosage_arr[i])

    psam = pd.read_csv(f_pgen[:-5] + ".psam", sep="\t", dtype={"#IID": np.str})

    return dosage_arr, psam


def read_global_ancestry(fname):
    return pd.read_csv(fname, sep="\t", header=None, dtype={0: np.str, 1: np.float})


if __name__ == "__main__":
    # CLI for debug only
    # read_rfmix(sys.argv[1])
    pass
