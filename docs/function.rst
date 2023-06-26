##############
Running HAMSTA
##############

***
SVD
***

.. code-block:: python

    hamsta import core, io, preprocess, utils

First, you need to compute the singular value decomposition of the local ancestry data. You can provide either a local ancestry file or a local ancestry correlation matrix (LAD) that contains the markers in your admixture mapping summary statistics.

1) If you proceed with local ancestry, you can use the following code to read local ancestry calls from an rfmix output. The code reads the ancestry "HCB" from local ancestry file ``example.fb.tsv``. The array A should be of dimension (#marker, #sample) and element ``A[i,j]`` are the number of copies of HCB ancestry the i-th individual has at j-th marker. 


    .. code-block:: python

        A, A_sample = io.read_rfmixfb("example.fb.tsv", "HCB")
        U, S = preprocess.SVD(A=A, Q=None, k=None)



   Optional argument ``Q`` (# sample, # covariates) for regressing out covariates from local acnestry, ``k`` for specifying number of components in truncated SVD. 

2) If you choose to decompose an LAD matrix, you can store the matrix in a numpy ndarray and pass it to ``preprocess.SVD``.


    .. code-block:: python

        U, S = preprocess.SVD(LAD=LAD, k=args.k)


*********
Inference
*********

In the inference step, you will


    .. code-block:: python

        ham = core.HAMSTA(S_thres=S_THRES)
        ham.fit(rotated_Z=Z_, S=S_, N=N, M=M, jackknife=True, intercept_design=intercept_design)
