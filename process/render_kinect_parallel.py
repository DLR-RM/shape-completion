import logging
import os
import shutil
from argparse import Namespace
from pathlib import Path
from time import time

from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

from utils import (tqdm_joblib, setup_logger, set_log_level, eval_input, disable_multithreading, resolve_out_dir,
                   save_command_and_args_to_file)
from .render_kinect import get_argument_parser, render

logger = setup_logger(__name__)


def run(in_path: Path, args: Namespace):
    start = time()
    logger.debug(f"Processing file {in_path}.")
    try:
        for shard in range(args.n_shards):
            shard = None if args.n_shards == 1 else shard
            out_dir = resolve_out_dir(in_path, args.in_dir, args.out_dir, shard) if args.out_dir else in_path.parent
            render(in_path, out_dir, args)
    except Exception as e:
        logger.exception(e)
        if args.remove:
            logger.warning(f"Exception occurred. Removing {out_dir}.")
            shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "lock").unlink(missing_ok=True)
    logger.debug(f"Runtime: {time() - start:.2f}s.\n")


def main():
    parser = get_argument_parser()
    parser.add_argument("in_dir", type=Path, help="Path to input directory.")
    parser.add_argument("--in_format", type=str, default=".off", help="Input file format.")
    parser.add_argument("--recursion_depth", type=int, help="Depth of recursive glob pattern matching.")
    parser.add_argument("--sort", action="store_true", help="Sort files before processing.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--n_shards", type=int, default=1,
                        help="Number of shards to split the data into.")
    args = parser.parse_args()

    save_command_and_args_to_file(args.out_dir / "command.txt", args)

    # Disable multithreading if multiprocessing is used.
    if args.n_jobs > 1:
        disable_multithreading()

    # Check that fix or remove are only set when check is set.
    if args.fix or args.remove:
        assert args.check, "Fix or remove can only be set when check is set."

    # Check that fix and remove are not set at the same time.
    assert not (args.fix and args.remove), "Fix and remove cannot be set at the same time."

    if args.verbose:
        set_log_level(logging.DEBUG)

    if not args.show:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    files = eval_input(args.in_dir, args.in_format, args.recursion_depth, args.sort)
    start = time()
    desc = "Checking" if args.check else "Rendering"
    if args.fix:
        desc += " & Fixing"
    if args.remove:
        desc += " & Removing"
    logger.debug(desc)
    with tqdm_joblib(tqdm(desc=desc, total=len(files), disable=args.verbose)):
        Parallel(n_jobs=1 if args.verbose else min(args.n_jobs, len(files)),
                 verbose=args.verbose)(delayed(run)(file, args) for file in files)
    logger.debug(f"Total runtime: {time() - start:.2f}s.")


if __name__ == "__main__":
    main()
