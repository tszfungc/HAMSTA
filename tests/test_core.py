import numpy as np
from scipy import stats

from hamsta import HAMSTA


def test_fit():
    S = np.load("tests/testdata/example.S.npy")
    S = S[S > 1e-3]
    h2a = 0.05
    M = 8000

    ham = HAMSTA()

    rotated_Z = stats.norm.rvs(loc=0.0, scale=np.sqrt(h2a / M * S ** 4 + S ** 2))
    ham.fit(rotated_Z=rotated_Z, S=S, M=M, jackknife=False)

    print(ham.to_dataframe())
