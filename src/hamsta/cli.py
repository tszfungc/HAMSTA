import argparse
import logging
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import linalg

from hamsta import __version__, core, io, preprocess, utils

_logger = logging.getLogger(__name__)
# logging.basicConfig(format="%(asctime)s | %(message)s")


def setup_logging(loglevel):
    """Setup basic logging
    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "%(asctime)s | %(message)s"
    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_parser():
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
        default=logging.WARN,
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
    preprocess_parser.add_argument(
        "--rfmixfb",
        nargs=2,
        help="Input local ancestry in rfmix .fb.tsv format, two args require, (filepath, ancestry)",  # noqa: E501
    )
    preprocess_parser.add_argument(
        "--zarr",
        nargs=2,
        help="Input local ancestry in zarr format storing an Xarray dataset, two args require, (filepath, ancestry)",  # noqa: E501
    )
    preprocess_parser.add_argument(
        "--nc",
        nargs=2,
        help="Input local ancestry in netcdf format storing an Xarray dataset, two args require, (filepath, ancestry)",  # noqa: E501
    )
    preprocess_parser.add_argument(
        "--global-ancestry", help="Path to global ancestry file in rfmix.Q format"
    )
    # preprocess_parser.add_argument("--LADmat", help="Path to LAD matrix")
    # preprocess_parser.add_argument("--N", help="Number of individuals", type=float)
    preprocess_parser.add_argument("--out", help="output prefix")
    preprocess_parser.add_argument(
        "--keep", help="file of a list of individual to keep"
    )
    preprocess_parser.add_argument(
        "--k", help="Number of singular values to compute", type=int
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
    infer_parser.add_argument("--N", help="Number of individuals", type=int)
    infer_parser.add_argument("--num-blocks", help="Number of jackknife blocks", type=int, default=10)
    infer_parser.add_argument(
        "--thres",
        help="whether significance threshold is estimated",
        type=bool,
        default=False,
    )
    infer_parser.add_argument("--out", help="output prefix", default=sys.stdout)
    infer_parser.set_defaults(func=infer_main)

    # organize parser
    main_parser = argparse.ArgumentParser(parents=[top_parser])
    subparsers = main_parser.add_subparsers(description="Subcommand")
    subparsers.add_parser("preprocess", parents=[preprocess_parser, top_parser])
    subparsers.add_parser("infer", parents=[infer_parser, top_parser])

    return main_parser


def pprocess_main(args):
    _logger.warning("Preprocess data, generate SVD results")
    # read global
    if args.global_ancestry is not None:
        _logger.warning("Reading global ancestry...")
        Q = io.read_global_ancestry(
            fname=args.global_ancestry, sample_colname="#sample", skiprows=1
        )
    else:
        Q = None

    # read local
    if args.rfmixfb is not None:
        _logger.warning("Reading local ancestry from rfmix .fb.tsv output...")
        A, A_sample = io.read_rfmixfb(*args.rfmixfb)
    elif args.nc is not None:
        _logger.warning("Reading local ancestry in netcdf4...")
        A, A_sample = io.read_nc(*args.nc)
    elif args.zarr is not None:
        _logger.warning("Reading local ancestry in zarr...")
        A, A_sample = io.read_zarr(*args.zarr)
    else:
        raise RuntimeError("No input local ancestry")

    # read local - logging
    _logger.warning(
        f"Found {A.shape[1]} individuals & {A.shape[0]} markers in local ancestry file"
    )
    num_indv_0 = A.shape[1]

    # read keep
    if args.keep is not None:
        _logger.warning("filtering to keep samples...")
        keep = pd.read_csv(args.keep, sep="\t")["#IID"].values.astype(object)
        keep = (A_sample[np.in1d(A_sample["sample"], keep)].merge(Q))["sample"]

        # filter local ancestry samples
        A_sel = np.in1d(A_sample["sample"], keep)
        A, A_sample = A[:, A_sel], A_sample[A_sel]

    num_indv_1 = A.shape[1]
    _logger.warning(f"{num_indv_0 - num_indv_1} individuals were filtered")

    if Q is not None:
        # sort global ancestry to local ancestry's order
        Q = A_sample.merge(Q)
        assert np.all(A_sample["sample"].values == Q["sample"].values)
        # astype jnp ndarray
        Q = jnp.array(Q.iloc[:, 1:2])

    # SVD
    _logger.warning(f"Running SVD on {A.shape[1]} individuals and {A.shape[0]} markers")
    U, S = preprocess.SVD(A=A, Q=Q, k=args.k)

    if args.out is not None:
        np.save(args.out + ".SVD.U.npy", U)
        np.save(args.out + ".SVD.S.npy", S)
        # np.save(outprefix + ".SVD.SDpj.npy", SDpj)
        _logger.warning("SVD out saved to " + args.out + ".SVD.*.npy")
        _logger.warning(f"output dimension: U {U.shape} S {S.shape}")


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
        M = Z.shape[0]
        U_, S_ = np.load(args.svd[0]), np.load(args.svd[1])
        Z_ = core.rotate(U=U_, S=S_, Z=Z, residual_var=RESIDUAL_VAR)
        intercept_design = utils.make_intercept_design(Z_.shape[0], binsize=BIN_SIZE)

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
            S = S * jnp.sqrt(args.N)
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

    ham.fit(rotated_Z=Z_, S=S_, M=M, jackknife=True, intercept_design=intercept_design, num_blocks=args.num_blocks)

    if ham.result["p_intercept"] < 0.05:
        thres_var = np.max(ham.result["parameter"][1:])
    else:
        thres_var = ham.result["mean_intercept"]

    thres = None
    if args.thres:
        burden_list = []
        for svd_line in open(args.svd_chr, "r"):
            U_f, S_f = svd_line.strip().split("\t")
            U, S = np.load(U_f), np.load(S_f)
            S = S * np.sqrt(args.N)
            intercept = np.repeat(thres_var, S.shape[0])
            thres = ham.compute_thres(fwer=0.05, U=U, S=S, intercept=intercept)
            burden_list.append(0.05 / thres)

        thres = 0.05 / sum(burden_list)

    res = ham.to_dict()
    res.update({"thres": [thres]})

    # res.to_csv(args.out, sep="\t", index=None)
    out_f = open(args.out, "w")
    for k in res:
        print(f"{k}\t{res[k]}", file=out_f)

    return 0


def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    setup_logging(args.loglevel)

    _logger.warning("Program starts")
    _logger.debug("DEBUG logging in on")

    if "func" in args:
        args.func(args)

    _logger.info("Program Finish")


def run():
    # CLI entry point
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
