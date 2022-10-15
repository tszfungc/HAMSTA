==========
Quickstart
==========

Command line interface
======================

Singular value decomposition of the local ancestry data
*******************************************************

.. code-block:: bash

    hamsta preprocess \
     --rfmixfb example.fb.tsv HCB \
     --global-ancestry example.rfmix.Q \
     --out example


The command will create files

::

    example.U.npy
    example.S.npy

Running HAMSTA
**************

.. code-block:: bash

    hamsta infer \
     --sumstat example.pheno.glm.linear \
     --svd example.U.npy example.S.npy \
     --out example.hamsta_result.txt

The result will be written to ``example.hamsta_result.txt`` in tab-delimited format

Python api
==========

Coming soon
