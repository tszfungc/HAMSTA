import numpy as np
from scipy import stats

from hamsta import HAMSTA


def test_fit():
    U = np.load(".pytest_cache/example.SVD.U.npy")
    S = np.load(".pytest_cache/example.SVD.S.npy")
    S = S[S > 1e-3]
    U = U[S > 1e-3]
    h2a = 0.05
    M = 8000

    ham = HAMSTA()

    rotated_Z = stats.norm.rvs(loc=0.0, scale=np.sqrt(h2a / M * S ** 4 + S ** 2))
    ham.fit(rotated_Z=rotated_Z, U=U, S=S, M=M, jackknife=False, est_thres=True)

    print(ham.to_dataframe())
