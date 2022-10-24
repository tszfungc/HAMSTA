==========
Quickstart
==========

Command line interface
======================


Example
-------

.. code-block:: bash

    hamsta preprocess \
     --rfmixfb example.fb.tsv HCB \
     --global-ancestry example.rfmix.Q \
     --out example


The command will create files

::

    example.U.npy
    example.S.npy

.. code-block:: bash

    hamsta infer \
     --sumstat example.pheno.glm.linear \
     --svd example.U.npy example.S.npy \
     --out example.hamsta_result.txt

The result will be written to ``example.hamsta_result.txt`` in tab-delimited format

Single value decomposition of local ancestry data
-------------------------------------------------

.. argparse::
    :module: hamsta.cli
    :func: parse_args
    :prog: hamsta
    :path: preprocess


.. note::

    Expect the following structure when Xarray dataset is used.

    ::

        <xarray.Dataset>
        Dimensions:           (ancestry: 2, marker: 8, sample: 39, ploidy: 2)
        Coordinates:
          * ancestry          (ancestry) <U3 'HCB' 'JPT'
          * marker            (marker) uint32 1 6 12 20 25 31 36 43
          * ploidy            (ploidy) int8 0 1
          * sample            (sample) <U6 'HCB182' 'HCB190' ... 'JPT266' 'JPT267'
        Data variables:
            genetic_position  (marker) float32 ...
            locanc            (marker, sample, ploidy, ancestry) float32 ...



Running HAMSTA
--------------

.. argparse::
    :module: hamsta.cli
    :func: parse_args
    :prog: hamsta
    :path: infer



Python api
==========


Coming soon
