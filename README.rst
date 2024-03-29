.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/HAMSTA.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/HAMSTA
    .. image:: https://readthedocs.org/projects/HAMSTA/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://HAMSTA.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/HAMSTA/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/HAMSTA
    .. image:: https://img.shields.io/pypi/v/HAMSTA.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/HAMSTA/
    .. image:: https://img.shields.io/conda/vn/conda-forge/HAMSTA.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/HAMSTA
    .. image:: https://pepy.tech/badge/HAMSTA/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/HAMSTA
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/HAMSTA

    .. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
        :alt: Project generated with PyScaffold
        :target: https://pyscaffold.org/


.. image:: https://github.com/tszfungc/HAMSTA/actions/workflows/docdeploy.yml/badge.svg
    :alt: Documentation status
    :target: https://tszfungc.github.io/HAMSTA/

|

======
HAMSTA
======


    Heritability estimation from Admixture Mapping Summary STAtistics



HAMSTA is a python package that estimate heritability explained by local ancestry using summary statistics from admixture mapping studies. It also quantifies inflation in test statistics that is not contributed by local ancestry effects, and determines significance threshold for admixture mapping.


Documentation
=============

For API and detailed documentation, please check out http://tszfungc.github.io/HAMSTA/

Installation
============

.. code-block:: bash

    git clone https://github.com/tszfungc/HAMSTA.git
    cd HAMSTA
    pip install -r requirement.txt
    python setup.py install

Example
=======

Perform SVD on local ancestry and regress out global ancestry in RFMIX output format.

.. code-block:: bash

    hamsta preprocess \
        --rfmixfb example.fb.tsv AFR \
        --global-ancestry example.rfmix.Q \
        --out example

Estimate heritability explained by local ancestry using admixture mapping results (e.g. PLINK2 glm) and SVD results.

.. code-block:: bash

    hamsta infer \
        --sumstat example.pheno.glm.linear \
        --svd example.U.npy example.S.npy \
        --out example.hamsta_result.txt

Reference
=========

Chan, T.F., Rui, X., Conti, D.V., Fornage, M., Graff, M., Haessler, J., Haiman, C., Highland, H.M., Jung, S.Y., Kenny, E., et al. (2023). Estimating heritability explained by local ancestry and evaluating stratification bias in admixture mapping from summary statistics. bioRxiv. `10.1101/2023.04.10.536252 <https://doi.org/10.1101/2023.04.10.536252>`_
