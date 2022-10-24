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

::

    usage: hamsta preprocess [-h] [--rfmixfb RFMIXFB RFMIXFB] [--zarr ZARR ZARR] [--nc NC NC] [--global-ancestry GLOBAL_ANCESTRY] [--N N] [--out OUT] [--keep KEEP] [--k K] [--version] [-v] [-vv]

    options:
      -h, --help            show this help message and exit
      --rfmixfb RFMIXFB RFMIXFB
                            rfmix output .fb.tsv, two args require, (filepath, ancestry)
      --zarr ZARR ZARR      Xarray dataset in zarr, two args require, (filepath, data_var)
      --nc NC NC            Xarray dataset in netcdf, two args require, (filepath, data_var)
      --global-ancestry GLOBAL_ANCESTRY
                            Path to rfmix.Q
      --N N                 Number of individuals
      --out OUT             output prefix
      --keep KEEP           list of individual to keep
      --k K                 Number of components to compute
      --version             show program's version number and exit
      -v, --verbose         set loglevel to INFO
      -vv, --very-verbose   set loglevel to DEBUG


.. argparse::
    :module: hamsta.cli
    :func: parse_args
    :prog: hamsta
    :path: preprocess

Python api
==========

Coming soon
