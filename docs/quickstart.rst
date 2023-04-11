==========
Quickstart
==========

Command line interface
======================


Example
-------

Step 1

.. code-block:: bash

    hamsta preprocess \
     --rfmixfb example.fb.tsv AFR \
     --global-ancestry example.rfmix.Q \
     --out example


The command will create files for the ``U`` and ``S`` in ``A' = USV'``, where the shape of ``A'`` is (marker, sample)

::

    example.U.npy
    example.S.npy

Step 2

.. code-block:: bash

    hamsta infer \
     --sumstat example.pheno.glm.linear \
     --svd example.U.npy example.S.npy \
     --out example.hamsta_result.txt

The result will be written to ``example.hamsta_result.txt``

Single value decomposition of local ancestry data
-------------------------------------------------

.. argparse::
    :module: hamsta.cli
    :func: get_parser
    :prog: hamsta
    :path: preprocess


.. note::

    If Xarray dataset is used, the following structure is expected

    ::

        <xarray.Dataset>
        Dimensions:           (marker: 800, sample: 3000, ploidy: 2, ancestry: 2)
        Coordinates:
          * marker            (marker) uint32 15309459 15343272 ... 50702360 50743879
          * sample            (sample) <U7 'msp1' 'msp2' 'msp3' ... 'msp2999' 'msp3000'
          * ploidy            (ploidy) int8 0 1
          * ancestry          (ancestry) <U3 'AFR' 'EUR'
        Data variables:
            locanc            (marker, sample, ploidy, ancestry) float32 1.0 0.0 ... 0.0
            genetic_position  (marker) float64 0.0 0.009117 0.01411 ... 73.63 73.78 73.9



Running HAMSTA
--------------

.. argparse::
    :module: hamsta.cli
    :func: get_parser
    :prog: hamsta
    :path: infer



Python api
==========


Coming soon
