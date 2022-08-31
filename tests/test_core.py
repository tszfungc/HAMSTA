import numpy as np
from scipy import stats
from tqdm import tqdm

from hamsta import HAMSTA


def test_fit():
    S = np.load("tests/testdata/example.S.npy")
    h2a = 0.001
    M = 120000

    ham = HAMSTA()

    for i in tqdm(range(10)):
        rotated_Z = stats.norm.rvs(loc=0.0, scale=np.sqrt(h2a / M * S ** 4 + S ** 2))
        ham.fit(rotated_Z=rotated_Z, S=S, M=M, jackknife=False)
        print(vars(ham))

    assert True
