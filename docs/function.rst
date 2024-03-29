##############
Running HAMSTA
##############

***
SVD
***

.. code-block:: python

    hamsta import core, io, preprocess, utils

First, you need to compute the singular value decomposition of the local ancestry data. You can provide either a local ancestry file or a local ancestry correlation matrix (LAD) that contains the markers in your admixture mapping summary statistics.
When optional argument ``k`` is set, truncated SVD is performed for the specified number of components. Otherwise, full SVD will be performed.


1) If you proceed with local ancestry, you can use the following code to read local ancestry calls from an rfmix output. The code reads the ancestry "HCB" from local ancestry file ``example.fb.tsv``. The array ``A`` should be of dimension (#marker, #sample) and element ``A[i,j]`` are the number of copies of HCB ancestry present in the i-th marker of the j-th individual. The singular values will be scaled according to the sample size.


    .. code-block:: python

        A, A_sample = io.read_rfmixfb(fname="example.fb.tsv", ancestry="HCB")
        U, S = preprocess.SVD(A=A, Q=None, k=None)



   Optional argument ``Q`` (# sample, # covariates) for regressing out covariates from local acnestry.


2) If you choose to decompose an LAD matrix, you can store the matrix in a numpy ndarray and pass it to ``LAD`` in :func:`.preprocess.SVD`.


    .. code-block:: python

        U, S = preprocess.SVD(LAD=LAD, k=None)


*********
Inference
*********

In the inference step, we can create a :class:`.HAMSTA` object and use :func:`.HAMSTA.fit` to fit the processed statistics and singular values. Arguments ``Z`` and ``N`` are the admixture mapping test statistics and sample size. ``intercept_design`` is a design matrix (number of singular values, number of distinct intercept) for assign intercepts to each rotated Z. For single intecept, we can set ``intercept_design = np.ones((S.shape[0], 1))``. We also provide a function ``utils.make_intercept_design(S.shape[0], bin_size=500)`` to specify a new intercept for every 500 rotated Z for example.

    .. code-block:: python

        ham = core.HAMSTA()
        ham.fit(Z=Z, U=U, S=S, N=N, jackknife=True, intercept_design=intercept_design, est_thres=True)

To combine input from multiple chromosomes, apply ``np.concatenate`` on the list of ``Z`` or ``S`` and ``scipy.linalg.block_diag`` on ``U``. 

You can also rotate Z for each chromosome and pass them to argument ``rotated_Z``.

    .. code-block:: python

        rotated_Z_list = [core.rotate(U=U_list[i], S=S_list[i], Z=Z_list[i], N=N) for i in range(22)]
        rotated_Z = np.concatenate(rotated_Z_list)
        S = np.concatenate(S_list)

        ham = core.HAMSTA()
        ham.fit(rotated_Z=rotated_Z,S=S,M=M,jackknife=True,intercept_design=intercept_design,N=N)
    



This output the results in a dictionary ``ham.result`` containing h2gamma and intercepts estimates and their standard errors, followed by the log-likelihood for h2gamma=0 (h0), single intercept (h1_sintcpt) and multiple intercepts (h1_mintcpt) and the p-values for testing for h2gamma and multiple intercepts. Based on the LAD and estimated intercept, a significance threshold corresponding to family-wise error rate of 0.05 is provided.


Example result:
::

    h2gamma     [0.03280452]
    h2gamam_SE  [2.05591325e-12]
    intercepts      [0.97167965]
    intercepts_SE   [1.84471781e-07]
    mean_intercept  0.9716796462662657
    h0      -27125.71904267748
    h1_sintcpt      -26519.959294116295
    h1_mintcpt      -26505.393609470648
    p_h2gamma   0.0000e+00
    p_intercept     8.7520e-01
    thres   0.0001467017385257119
