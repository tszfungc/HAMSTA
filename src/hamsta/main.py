"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = hamsta.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!

References:
    - https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

from hamsta import __version__, estimation_jackknife, io, utils

__author__ = "tszfungc"
__copyright__ = "tszfungc"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from hamsta.skeleton import fib`,
# when using this Python module as a library.


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    topparser = argparse.ArgumentParser(
        description="Heritability Estimation from Admixture Mapping Summary Statistics"
    )
    parser_ = topparser.add_subparsers(help='Choose subcommand')

    # infer
    parsera = parser_.add_parser('infer', help='Inference')
    parsera.add_argument(
        "--version",
        action="version",
        version="HAMSTA {ver}".format(ver=__version__),
    )

    parsera.add_argument("sumstat", help="Input filename of admixture mapping results")
    parsera.add_argument("--rsumstat", help="Input filename of rotated Z")

    parsera.add_argument("--svdprefix", help="Prefix of the SVD results")
    parsera.add_argument("--svdprefix-chr", help="Prefix of the per chr SVD results")

    parsera.add_argument("--rfmixprefix", help="Prefix of the rfmix results")

    parsera.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel from INFO to DEBUG",
        default=logging.INFO,
        action="store_const",
        const=logging.DEBUG,
    )
    parsera.set_defaults(func=infer_main)


    # preprocess
    parserb = parser_.add_parser('pprocess', help='pre-process')
    parserb.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel from INFO to DEBUG",
        default=logging.INFO,
        action="store_const",
        const=logging.DEBUG,
    )
    parserb.add_argument("--pgen", help='Path to pgen')
    parserb.add_argument("--global-ancestry", help='Path to rfmix.Q')
    parserb.add_argument("--out", help='output prefix')
    parserb.set_defaults(func=pprocess_main)


    return topparser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper  in a CLI fashion"""
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    _logger.info("Program Starts")

    # Preview the argparse results
    args_dict = vars(args)
    parse_prtout = "\n".join([f"\t{i}: {args_dict[i]}" for i in args_dict])
    _logger.info(
        f"""
CLI called
{' '.join(sys.argv)}

Arguments parsed
---------------
{parse_prtout}
"""
    )

    # main func for subcommands
    args.func(args)

def pprocess_main(args):
    # read rfmix output
    #U, S, SDpj = io.read_rfmix_N_SVD(args.rfmixprefix
    Q = io.read_global_ancestry(args.global_ancestry)
    A = io.read_pgen(args.pgen)

    utils.SVD(A, Q, outprefix=args.out)



def infer_main(args):

    # main procedures
    # sumstat_df = io.read_sumstat(args.sumstat)
    # add Z or rotated Z
    # Z = sumstat_df['T_STAT'].values
    # sumstat_df = io.read_sumstat(args.sumstat)
    Z = io.read_sumstat(args.sumstat)
    _logger.info(
        f"""
Read sumstat; Number of markers: {Z.shape[0]}
    """
    )

    # if Z is supplied, rotated Z

    # read SVD prefix
    if args.svdprefix is not None:
        U, S, SDpj = io.read_SVD(args.svdprefix)

    if args.svdprefix_chr is not None:
        U, S, SDpj = io.read_SVD_chr(args.svdprefix_chr)

    _logger.info(
        f"""
Read SVD; Number of markers: {U.shape}
    """
    )

    assert S.shape[0] == Z.shape[0], "Mismatch dimension"
    #print(sum(S>1e-3))
    #U = U[S>1e-3]
    #SDpj = SDpj[S>1e-3]
    #Z = Z[S>1e-3]
    #S = S[S>1e-3]
    rotated_z = utils.rotate_Z(U, SDpj, Z)

    # After having M, S, rotated Z
    _logger.info("Summary stat rotated")
    estimation_jackknife.run(M=U.shape[0], S=S, rotated_z=rotated_z,
                             binsize=rotated_z.shape[0])

    _logger.info("Program ends")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m hamsta.skeleton 42
    #
    run()
