import logging
import shutil
import time
from argparse import ArgumentParser
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from utils import (
    disable_multithreading,
    log_optional_dependency_summary,
    resolve_out_dir,
    set_log_level,
    setup_logger,
    suppress_known_optional_dependency_warnings,
    tqdm_joblib,
)

logger = setup_logger(__name__)


def _read_dataset_bytes(entry: h5py.Dataset) -> bytes:
    payload = entry[()]
    if isinstance(payload, np.void):
        return payload.tobytes()
    if isinstance(payload, np.ndarray):
        return payload.tobytes()
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    raise TypeError(f"Unsupported HDF5 payload type: {type(payload)!r}")


def save_as_binary(hdf5_file: h5py.File, file_path: Path, group_name: str | None = None) -> None:
    """Reads a file and saves it as a binary blob in the HDF5 file under the specified group and file name."""
    with file_path.open("rb") as file:
        data = file.read()
    if group_name is None:
        hdf5_file.create_dataset(file_path.name, data=np.void(data))
    else:
        group = hdf5_file.create_group(group_name) if group_name not in hdf5_file else hdf5_file[group_name]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"Expected HDF5 group at '{group_name}', found {type(group)!r}")
        group.create_dataset(file_path.name, data=np.void(data))


def save_binary_hdf5(obj_path: Path, out_dir: Path | None = None) -> None:
    logger.debug(f"Processing item {obj_path}.")
    if out_dir is None:
        hdf5_path = obj_path.with_suffix(".hdf5")
    else:
        out_dir = resolve_out_dir(obj_path, obj_path.parent.parent, out_dir)
        hdf5_path = out_dir / obj_path.with_suffix(".hdf5").name
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "w") as hdf5_file:
        for dir_name in ["depth", "normal", "kinect", "samples"]:
            dir_path = obj_path / dir_name
            if dir_path.is_dir():
                for file_path in dir_path.iterdir():
                    save_as_binary(hdf5_file, file_path, dir_name)

        for file_name in ["model.off", "parameters.npz"]:
            file_path = obj_path / file_name
            if file_path.is_file():
                save_as_binary(hdf5_file, file_path)


def load_binary_hdf5(hdf5_path: Path, out_dir: Path | None = None) -> None:
    logger.debug(f"Processing item {hdf5_path}.")
    if out_dir is None:
        obj_path = hdf5_path.parent / hdf5_path.stem
    else:
        out_dir = resolve_out_dir(hdf5_path, hdf5_path.parent.parent, out_dir)
        obj_path = out_dir / hdf5_path.stem
        obj_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        for dir_name in ["depth", "normal", "kinect", "samples"]:
            if dir_name in hdf5_file:
                dir_path = obj_path / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                dir_data = hdf5_file[dir_name]
                if not isinstance(dir_data, h5py.Group):
                    raise TypeError(f"Expected HDF5 group at '{dir_name}', found {type(dir_data)!r}")
                for file_name, dataset in dir_data.items():
                    if not isinstance(dataset, h5py.Dataset):
                        raise TypeError(f"Expected HDF5 dataset for '{dir_name}/{file_name}', found {type(dataset)!r}")
                    file_path = dir_path / file_name
                    data = BytesIO(_read_dataset_bytes(dataset))
                    with file_path.open("wb") as f:
                        f.write(data.read())

        for file_name in ["model.off", "parameters.npz"]:
            file_path = obj_path / file_name
            if file_name in hdf5_file:
                dataset = hdf5_file[file_name]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError(f"Expected HDF5 dataset at '{file_name}', found {type(dataset)!r}")
                data = BytesIO(_read_dataset_bytes(dataset))
                with file_path.open("wb") as f:
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
        val_set = files[n_train : n_train + n_val]
        test_set = files[n_train + n_val :]

        logger.debug(f"Writing splits to {out_dir}.")
        with open(out_dir / "train.lst", "w") as f:
            f.write("\n".join(train_set))

        with open(out_dir / "val.lst", "w") as f:
            f.write("\n".join(val_set))

        with open(out_dir / "test.lst", "w") as f:
            f.write("\n".join(test_set))

        with open(out_dir / "all.lst", "w") as f:
            f.write("\n".join(files))

        with open(out_dir / "train_val.lst", "w") as f:
            f.write("\n".join(train_set + val_set))

        with open(out_dir / "train_test.lst", "w") as f:
            f.write("\n".join(train_set + test_set))
        logger.debug(f"Done splitting {c}.")
    logger.debug("Done creating splits.")


def merge_splits(args: Any):
    """
    Combines the train.lst, val.lst, and test.lst files from a specific
    set of class directories into single, combined files.
    """
    logger.info("Starting merge process for specified ShapeNet classes.")

    base_path = args.in_dir
    # If out_dir is not specified, write the merged files to the input directory.
    out_path = args.out_dir if args.out_dir else base_path
    if args.out_dir:
        out_path.mkdir(parents=True, exist_ok=True)

    combined_data: defaultdict[str, list[str]] = defaultdict(list)
    split_names = ["train", "val", "test"]

    # Iterate over each class defined in TARGET_CLASSES
    for class_id in args.in_dir.iterdir():
        if not class_id.is_dir():
            continue

        class_dir = base_path / class_id
        logger.info(f"Processing class {class_id}")

        # Iterate over train.lst, val.lst, test.lst
        for split in split_names:
            list_file = f"{split}.lst"
            file_path = class_dir / list_file

            try:
                with file_path.open() as f:
                    # Read model IDs and prepend the class directory for identification
                    suffix = "model.obj"
                    if "v2" in str(class_dir).lower():
                        suffix = "models/model_normalized.obj"
                    lines = [f"{class_dir}/{line.strip()}/{suffix}\n" for line in f if line.strip()]
                    combined_data[split].extend(lines)
                    logger.debug(f"  - Read {len(lines)} entries from {list_file}")
            except FileNotFoundError:
                logger.warning(f"  - {list_file} not found in {class_dir}. Skipping this file.")

    logger.info("Finished reading all class files. Now writing combined files...")

    # Write the combined data to new train.lst, val.lst, and test.lst files
    for split, content in combined_data.items():
        if content:
            output_filepath = out_path / f"{split}_objs.txt"
            try:
                with output_filepath.open("w") as f:
                    f.writelines(content)
                logger.info(f"  - Successfully created '{output_filepath}' with {len(content)} total entries.")
            except OSError as e:
                logger.error(f"  - Could not write to {output_filepath}. Reason: {e}")
        else:
            logger.warning(f"  - No content to write for {split}.lst.")

    logger.info("Merge process completed.")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir", type=Path, help="Path to input directory.")
    parser.add_argument("task", type=str, choices=["split", "pack", "unpack", "merge"], help="Task to perform.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    suppress_known_optional_dependency_warnings()

    # Disable multithreading if multiprocessing is used.
    if args.n_jobs > 1:
        disable_multithreading()

    if args.verbose:
        set_log_level(logging.DEBUG)
    log_optional_dependency_summary(logger)

    if args.task == "split":
        create_splits(args)
    elif args.task == "merge":
        merge_splits(args)
    elif args.task in ["pack", "unpack"]:
        start = time.perf_counter()
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
            Parallel(n_jobs=1 if args.verbose else min(args.n_jobs, len(files)), verbose=args.verbose)(
                delayed(run)(file, args.out_dir) for file in files
            )
        logger.debug(f"Total runtime: {time.perf_counter() - start:.2f}s.")
