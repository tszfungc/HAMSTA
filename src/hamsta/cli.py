import argparse
import logging
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import linalg

from hamsta import __version__, core, io, preprocess, utils

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


def parse_args():
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
    preprocess_parser.add_argument(
        "--k", help="Number of components to compute", type=int
    )
    preprocess_parser.set_defaults(func=pprocess_main)

    # infer parser
    infer_parser = argparse.ArgumentParser(add_help=False)
    infer_parser.add_argument(
        "--sumstat", help="Input filename of admixture mapping results"
    )
    infer_parser.add_argument(
        "--sumstat-chr",
        help="file storing list of admixture mapping results",
    )
    infer_parser.add_argument("--svd", help="SVD results, U and S", nargs=2)
    infer_parser.add_argument(
        "--svd-chr", help="file storing list of SVD results, path to U and S each line"
    )
    # infer_parser.add_argument(
    #     "--k", help="number of singular values used in inference", type=int
    # )
    # infer_parser.add_argument("--N", help="number of individuals", type=int)
    infer_parser.add_argument("--out", help="output prefix", default=sys.stdout)
    infer_parser.set_defaults(func=infer_main)

    # organize parser
    main_parser = argparse.ArgumentParser(parents=[top_parser])
    subparsers = main_parser.add_subparsers(description="Subcommand")
    subparsers.add_parser("preprocess", parents=[preprocess_parser, top_parser])
    subparsers.add_parser("infer", parents=[infer_parser, top_parser])

    return main_parser


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
        A, A_sample = io.read_rfmixfb(*args.rfmixfb)
    elif args.nc is not None:
        A, A_sample = io.read_nc(*args.nc)
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

    if Q is not None:
        # sort global ancestry to local ancestry's order
        Q = A_sample.merge(Q)
        assert np.all(A_sample["sample"] == Q["sample"])
        # astype jnp ndarray
        Q = jnp.array(Q.iloc[:, 1:2])

    # SVD
    U, S = preprocess.SVD(A=A, Q=Q, k=args.k)

    if args.outprefix is not None:
        np.save(args.outprefix + ".SVD.U.npy", U)
        np.save(args.outprefix + ".SVD.S.npy", S)
        # np.save(outprefix + ".SVD.SDpj.npy", SDpj)
        _logger.info("SVD out saved to " + args.outprefix + ".SVD.*.npy")
        _logger.info(f"output dimension: U ({U.shape}) S ({S.shape})")


def infer_main(args):
    """main func for infer

    Args:
        args: argument include
             | sumstat or sumstat_chr
             | svd or svd_chr

    """

    Z_COLNAME = "T_STAT"
    S_THRES = 1.0
    BIN_SIZE = 500
    RESIDUAL_VAR = 1.0

    if args.sumstat is not None:
        Z = io.read_sumstat(args.sumstat, Z_colname=Z_COLNAME)
        U, S = np.load(args.svd[0]), np.load(args.svd[1])
        intercept_design = utils.make_intercept_design(Z.shape[0], binsize=BIN_SIZE)

    elif args.sumstat_chr is not None and args.svd_chr is not None:
        Z_list, intercept_design_list, S_list = [], [], []
        M = 0
        for sumstat_line, svd_line in zip(
            open(args.sumstat_chr, "r"), open(args.svd_chr, "r")
        ):
            Z = io.read_sumstat(sumstat_line.strip(), Z_colname="T_STAT")
            M += Z.shape[0]

            U_f, S_f = svd_line.strip().split("\t")
            U, S = np.load(U_f), np.load(S_f)
            S_list.append(S)

            rotated_Z = core.rotate(U=U, S=S, Z=Z, residual_var=RESIDUAL_VAR)
            Z_list.append(rotated_Z)
            intercept_design_list.append(
                utils.make_intercept_design(rotated_Z.shape[0], binsize=BIN_SIZE)
            )
        Z_ = jnp.concatenate(Z_list)
        S_ = jnp.concatenate(S_list)
        intercept_design = linalg.block_diag(*intercept_design_list)

    else:
        raise RuntimeError("Insufficient arguments")

    ham = core.HAMSTA(S_thres=S_THRES)

    ham.fit(rotated_Z=Z_, S=S_, M=M, jackknife=True, intercept_design=intercept_design)

    if ham.result["p_intercept"] < 0.05:
        thres_var = np.max(ham.result["parameter"][1:])
    else:
        thres_var = ham.result["mean_intercept"]

    burden_list = []
    for svd_line in open(args.svd_chr, "r"):
        U_f, S_f = svd_line.strip().split("\t")
        U, S = np.load(U_f), np.load(S_f)
        intercept = np.repeat(thres_var, S.shape[0])
        thres = ham.compute_thres(fwer=0.05, U=U, S=S, intercept=intercept)
        burden_list.append(0.05 / thres)

    thres = 0.05 / sum(burden_list)
    res = ham.to_dataframe()
    res.update({"thres": [thres]})

    res.to_csv(args.out, sep="\t", index=None)

    return 0


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
