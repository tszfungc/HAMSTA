import argparse
import logging
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd

from hamsta import __version__, io, preprocess

_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging
    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:\n%(message)s"
    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def parse_args(args):
    # Define parental parsers for main and subcommand
    # template for main
    top_parser = argparse.ArgumentParser(add_help=False)
    top_parser.add_argument(
        "--version",
        action="version",
        version="HAMSTA {ver}".format(ver=__version__),
    )
    top_parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    top_parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    # preprocess parser
    preprocess_parser = argparse.ArgumentParser(add_help=False)
    # preprocess_parser.add_argument("--pgen", help="Path to pgen")
    # preprocess_parser.add_argument("--nc", help="Path to xarray nc")
    preprocess_parser.add_argument(
        "--rfmixfb",
        nargs=2,
        help="rfmix output .fb.tsv, two args require, (filepath, ancestry)",
    )
    preprocess_parser.add_argument(
        "--zarr",
        nargs=2,
        help="Xarray dataset in zarr, two args require, (filepath, data_var)",
    )
    preprocess_parser.add_argument(
        "--nc",
        nargs=2,
        help="Xarray dataset in netcdf, two args require, (filepath, data_var)",
    )
    preprocess_parser.add_argument("--global-ancestry", help="Path to rfmix.Q")
    # preprocess_parser.add_argument("--LADmat", help="Path to LAD matrix")
    preprocess_parser.add_argument("--N", help="Number of individuals", type=float)
    preprocess_parser.add_argument("--out", help="output prefix")
    preprocess_parser.add_argument("--keep", help="list of individual to keep")
    preprocess_parser.add_argument("--k", help="Number of components to compute")
    preprocess_parser.set_defaults(func=pprocess_main)

    # infer parser
    infer_parser = argparse.ArgumentParser(add_help=False)
    infer_parser.add_argument(
        "--sumstat", help="Input filename of admixture mapping results"
    )
    infer_parser.add_argument(
        "--sumstat-chr",
        help="Input prefix and suffix of filename of admixture mapping results",
    )
    infer_parser.add_argument("--svd", help="Prefix of the SVD results", nargs=2)
    infer_parser.add_argument(
        "--svd-chr", help="Prefix and suffix of the per chr SVD results", nargs=2
    )
    # infer_parser.add_argument(
    #     "--k", help="number of singular values used in inference", type=int
    # )
    # infer_parser.add_argument("--N", help="number of individuals", type=int)
    infer_parser.set_defaults(func=infer_parser)

    # organize parser
    main_parser = argparse.ArgumentParser(parents=[top_parser])
    subparsers = main_parser.add_subparsers(description="Subcommand")
    subparsers.add_parser("preprocess", parents=[preprocess_parser, top_parser])
    subparsers.add_parser("infer", parents=[infer_parser, top_parser])

    return main_parser.parse_args(args)


def pprocess_main(args):

    # read global
    if args.global_ancestry is not None:
        Q = io.read_global_ancestry(
            fname=args.global_ancestry, sample_colname="#sample", skiprows=1
        )
    else:
        Q = None

    # read local
    if args.rfmixfb is not None:
        A, A_sample = io.raed_rfmixfb(*args.rfmixfb)
    elif args.nc is not None:
        A, A_sample = io.read_nc(*args.zarr)
    elif args.zarr is not None:
        A, A_sample = io.read_zarr(*args.zarr)
    else:
        raise RuntimeError("No input local ancestry")

    _logger.info(A_sample)

    # read keep
    if args.keep is not None:
        keep = pd.read_csv(args.keep, header=None)[0].values.astype(str)
        keep = (A_sample[np.in1d(A_sample["sample"], keep)].merge(Q))["sample"]

        # filter local ancestry samples
        A_sel = np.in1d(A_sample["sample"], keep)
        A, A_sample = A[:, A_sel], A_sample[A_sel]

    # sort global ancestry to local ancestry's order
    Q = A_sample.merge(Q)

    # astype jnp ndarray
    Q = jnp.array(Q.iloc[:, 1:-1])

    # SVD
    preprocess.SVD(A=A, Q=Q, k=args.k, outprefix=args.out)


def infer_main(args):
    pass


def main(args):
    args = parse_args(args)

    setup_logging(args.loglevel)

    if "func" in args:
        args.func(args)


def run():
    # CLI entry point
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
