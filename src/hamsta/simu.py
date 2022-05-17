import sys

import numpy as np
import pandas as pd
from scipy import stats

from hamsta import io

# input
pgen = sys.argv[1]
psam = pgen.replace('pgen', 'psam')
df = pd.read_csv(psam, sep='\t')

# hsq
hsq = np.float64(sys.argv[2])
A = io.read_pgen(pgen)
M, N = A.shape
A_std = stats.zscore(A, axis=1).T

# causal
n_signal = int(sys.argv[3])


# simu
for i in range(50):
    std_norm_effect = np.zeros(M)
    std_norm_effect[:n_signal] = np.random.normal(size=n_signal)
    np.random.shuffle(std_norm_effect)
    print(f"Number of non-zero effect: {np.sum(std_norm_effect!=0)}")

    std_norm_err = np.random.normal(size=(N, 1))

    cpnt_A = np.sqrt(hsq)*stats.zscore(A_std @ std_norm_effect[:, None], axis=0)
    cpnt_e = np.sqrt(1-hsq)*stats.zscore(std_norm_err, axis=0)
    pheno = (cpnt_A + cpnt_e).flatten()

    df[i] = pheno

df = df.drop('SEX', axis=1)
print(df)
df.to_csv(f"hsq{hsq:.3f}_{n_signal}.pheno", index=None, sep='\t')
