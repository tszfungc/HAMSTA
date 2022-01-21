import numpy as np
import pandas as pd
import pgenlib as pg
from scipy.linalg import block_diag
from tqdm import tqdm

from hamsta import utils


def read_sumstat(fname):
    df = pd.read_csv(fname, sep="\t")

    Z_colname = ["Z", "T_STAT", "STAT", "ZSCORE", "Z-SCORE", "GC_ZSCORE"]

    Z_series = df.iloc[:, np.in1d(df.columns, Z_colname)]
    assert Z_series.shape[1] == 1, "More than 1 columns of test statistcs found."

    return Z_series.values.squeeze()


def read_rfmix_N_SVD(fname):
    """Process  RFMIX output"""

    # Check file exists

    # Parse the header
    f = open(fname + ".fb.tsv", "r")
    line = next(f)
    npop = len(line.strip().split("\t")[1:])
    line = next(f)
    # sample_list_header = line.strip().split("\t")[4 :: npop * 2]
    # sample_list = list(map(lambda x: x.split(":::")[0], sample_list_header))

    # traw
    print("Reading content")
    A_mat = []
    for line in tqdm(list(f)):
        line_split = line.strip().split("\t")
        # pos = line_split[1]
        dosage = np.float_(line_split[4 :: npop * 2]) + np.float_(
            line_split[(4 + npop) :: npop * 2]
        )

        A_mat.append(dosage)

    A_mat = np.array(A_mat)

    # get Q
    Q = np.loadtxt(fname + ".rfmix.Q", usecols=1)

    print("Finish reading Shape:", end=" ")
    print(A_mat.shape)

    return utils.SVD(A_mat, Q, outprefix=fname)

def read_SVD_chr(svdprefix_chr):
    U_list = [np.load(f"{svdprefix_chr}.{i}.SVD.U.npy") for i in range(1, 23)]
    S_list = [np.load(f"{svdprefix_chr}.{i}.SVD.S.npy") for i in range(1, 23)]
    SDpj = [np.load(f"{svdprefix_chr}.{i}.SVD.SDpj.npy") for i in range(1, 23)]

    return block_diag(*U_list), np.concatenate(S_list), np.concatenate(SDpj)




def read_SVD(svdprefix):
    U = np.load(svdprefix + ".SVD.U.npy")
    S = np.load(svdprefix + ".SVD.S.npy")
    SDpj = np.load(svdprefix + ".SVD.SDpj.npy")

    return U, S, SDpj

def read_pgen(f_pgen):
    reader = pg.PgenReader(f_pgen.encode('utf-8'))
    dosage_arr = np.empty(
        (reader.get_variant_ct(),reader.get_raw_sample_ct()),
         dtype=np.float)

    print(f'Reading pgen file (n, p): ({dosage_arr.shape[1]}, {dosage_arr.shape[0]})')

    for i in range(dosage_arr.shape[0]):
        reader.read_dosages(i, dosage_arr[i])

    print(dosage_arr)

    return dosage_arr

def read_global_ancestry(fname):
    return np.loadtxt(fname, usecols=1)

if __name__ == "__main__":
    # CLI for debug only
    # read_rfmix(sys.argv[1])
    pass
