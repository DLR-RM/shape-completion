from typing import Any
from pathlib import Path
from argparse import ArgumentParser
import logging
from typing import Optional
from time import time
import shutil
from io import BytesIO

from joblib import Parallel, delayed, cpu_count
import h5py
from tqdm import tqdm
import numpy as np

from utils import setup_logger, set_log_level, tqdm_joblib, disable_multithreading, resolve_out_dir, save_mesh

logger = setup_logger(__name__)


def save_as_binary(hdf5_file: h5py.File, file_path: Path, group_name: Optional[str] = None) -> None:
    """Reads a file and saves it as a binary blob in the HDF5 file under the specified group and file name."""
    with file_path.open('rb') as file:
        data = file.read()
    if group_name is None:
        hdf5_file.create_dataset(file_path.name, data=np.void(data))
    else:
        group = hdf5_file.create_group(group_name) if group_name not in hdf5_file else hdf5_file[group_name]
        group.create_dataset(file_path.name, data=np.void(data))


def save_binary_hdf5(obj_path: Path, out_dir: Optional[Path] = None) -> None:
    logger.debug(f"Processing item {obj_path}.")
    if out_dir is None:
        hdf5_path = obj_path.with_suffix('.hdf5')
    else:
        out_dir = resolve_out_dir(obj_path, obj_path.parent.parent, out_dir)
        hdf5_path = out_dir / obj_path.with_suffix('.hdf5').name
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for dir_name in ['depth', 'normal', 'kinect', 'samples']:
            dir_path = obj_path / dir_name
            if dir_path.is_dir():
                for file_path in dir_path.iterdir():
                    save_as_binary(hdf5_file, file_path, dir_name)

        for file_name in ['model.off', 'parameters.npz']:
            file_path = obj_path / file_name
            if file_path.is_file():
                save_as_binary(hdf5_file, file_path)


def load_binary_hdf5(hdf5_path: Path, out_dir: Optional[Path] = None) -> None:
    logger.debug(f"Processing item {hdf5_path}.")
    if out_dir is None:
        obj_path = hdf5_path.parent / hdf5_path.stem
    else:
        out_dir = resolve_out_dir(hdf5_path, hdf5_path.parent.parent, out_dir)
        obj_path = out_dir / hdf5_path.stem
        obj_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for dir_name in ['depth', 'normal', 'kinect', 'samples']:
            if dir_name in hdf5_file:
                dir_path = obj_path / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                dir_data = hdf5_file[dir_name]
                for file in dir_data:
                    file_path = dir_path / file
                    data = BytesIO(np.frombuffer(dir_data[file][()], dtype=np.uint8))
                    with file_path.open('wb') as f:
                        f.write(data.read())

        for file_name in ['model.off', 'parameters.npz']:
            file_path = obj_path / file_name
            if file_name in hdf5_file:
                data = BytesIO(np.frombuffer(hdf5_file[file_name][()], dtype=np.uint8))
                with file_path.open('wb') as f:
                    f.write(data.read())


def create_splits(args: Any):
    logger.debug("Creating splits.")
    classes = [c.stem for c in args.in_dir.iterdir() if c.is_dir()]
    classes.sort()
    logger.debug(f"Found {len(classes)} classes.")
    for c in tqdm(classes, desc="Creating splits", disable=args.verbose):
        logger.debug(f"Processing class {c}.")
        in_dir = args.in_dir / c
        out_dir = args.out_dir / c if args.out_dir else args.in_dir / c
        files = [f.stem for f in in_dir.iterdir() if f.is_dir()]
        files.sort()
        logger.debug(f"Found {len(files)} files for class {c}.")

        n_total = len(files)
        n_val = int(args.val_size * n_total)
        n_test = int(args.test_size * n_total)
        n_train = n_total - n_val - n_test
        logger.debug(f"Splitting {n_total} files into {n_train} train, {n_val} val, {n_test} test.")

        train_set = files[:n_train]
        val_set = files[n_train:n_train + n_val]
        test_set = files[n_train + n_val:]

        logger.debug(f"Writing splits to {out_dir}.")
        with open(out_dir / 'train.lst', 'w') as f:
            f.write('\n'.join(train_set))

        with open(out_dir / 'val.lst', 'w') as f:
            f.write('\n'.join(val_set))

        with open(out_dir / 'test.lst', 'w') as f:
            f.write('\n'.join(test_set))

        with open(out_dir / 'all.lst', 'w') as f:
            f.write('\n'.join(files))

        with open(out_dir / 'train_val.lst', 'w') as f:
            f.write('\n'.join(train_set + val_set))

        with open(out_dir / 'train_test.lst', 'w') as f:
            f.write('\n'.join(train_set + test_set))
        logger.debug(f"Done splitting {c}.")
    logger.debug("Done creating splits.")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir", type=Path, help="Path to input directory.")
    parser.add_argument("task", type=str, choices=["split", "pack", "unpack"], help="Task to perform.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # Disable multithreading if multiprocessing is used.
    if args.n_jobs > 1:
        disable_multithreading()

    if args.verbose:
        set_log_level(logging.DEBUG)

    if args.task == "split":
        create_splits(args)
    elif args.task in ["pack", "unpack"]:
        start = time()
        run = save_binary_hdf5 if args.task == "pack" else load_binary_hdf5
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            taxonomy_file = args.in_dir / "taxonomy.json"
            if taxonomy_file.is_file():
                shutil.copy(taxonomy_file, args.out_dir / taxonomy_file.name)
        classes = [c.stem for c in args.in_dir.iterdir() if c.is_dir()]
        logger.debug(f"Found {len(classes)} classes.")
        files = list()
        for c in classes:
            in_dir = args.in_dir / c
            if args.task == "pack":
                files_c = [in_dir / f.stem for f in in_dir.iterdir() if f.is_dir()]
            else:
                files_c = [in_dir / f.name for f in in_dir.iterdir() if f.suffix == ".hdf5"]
            logger.debug(f"Found {len(files_c)} files for class {c}.")
            files.extend(files_c)
            if args.out_dir is not None:
                for file in in_dir.glob("*.lst"):
                    out_dir_c = args.out_dir / c
                    out_dir_c.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, out_dir_c / file.name)

        with tqdm_joblib(tqdm(desc="Converting", total=len(files), disable=args.verbose)):
            Parallel(n_jobs=1 if args.verbose else min(args.n_jobs, len(files)),
                     verbose=args.verbose)(delayed(run)(file, args.out_dir) for file in files)
        logger.debug(f"Total runtime: {time() - start:.2f}s.")
