import atexit
import copy
import hashlib
import os
import pickle
import sys
import time
from logging import DEBUG
from multiprocessing import Manager, shared_memory
from typing import Any, cast

import numpy as np
import psutil
from torch.utils.data import DataLoader, Dataset

from utils import DEBUG_LEVEL_1, setup_logger

from .shapenet import ShapeNet
from .transforms import apply_transforms

logger = setup_logger(__name__)


def debug_level_1(message: str) -> None:
    log_fn = getattr(logger, "debug_level_1", logger.debug)
    log_fn(message)


def debug_level_2(message: str) -> None:
    log_fn = getattr(logger, "debug_level_2", logger.debug)
    log_fn(message)


def log_memory_usage(stage: str):
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.debug(f"[{stage}] Memory Usage: {mem_info.rss / (1024**3):.2f} GB")


def generate_content_hash(data: dict[str, Any]) -> str:
    """Generates an MD5 hash for a dictionary containing cachable data."""
    hasher = hashlib.md5()
    # Sort items by key for consistent hash results
    sorted_items = sorted(data.items())

    for key, value in sorted_items:
        hasher.update(key.encode("utf-8"))
        if isinstance(value, np.ndarray):
            hasher.update(value.tobytes())
        else:
            try:
                # Use pickle for other types, ensure it's deterministic if possible
                hasher.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception as e:
                logger.warning(
                    f"Could not pickle value for key {key} during hash generation: {e}. Using str representation as fallback."
                )
                hasher.update(str(value).encode("utf-8"))  # Basic fallback
    return hasher.hexdigest()


class SharedDataset(Dataset):
    def __init__(
        self,
        dataset: ShapeNet,
        shared_dict: Any,
        shared_hash_map: Any | None = None,
        shared_arrays: dict[str, np.ndarray] | None = None,
        shared_memories: dict[str, shared_memory.SharedMemory] | None = None,
    ):
        super().__init__()
        if shared_arrays is None != shared_memories is None:
            raise ValueError("Either both or none of shared_arrays and shared_memories should be provided.")
        if shared_hash_map is not None and shared_arrays is not None:
            raise NotImplementedError("Shared hash map and shared arrays/memories not supported yet.")

        self.shared_dict = shared_dict
        self.shared_arrays = shared_arrays
        self.shared_memories = shared_memories
        self.shared_hash_map = shared_hash_map

        self.dataset = copy.deepcopy(dataset)
        if logger.isEnabledFor(DEBUG_LEVEL_1):
            debug_level_1(f"Cachable fields: {[f for f in dataset.fields.values() if f.cachable]}")
            debug_level_1(f"Non-cachable fields: {[f for f in dataset.fields.values() if not f.cachable]}")

        self.cachable_trafos = None
        self.non_cachable_trafos = None
        dataset_any = cast(Any, dataset)
        transforms_3d = getattr(dataset_any, "transforms_3d", None)
        if transforms_3d is not None:
            self.cachable_trafos = [t for t in transforms_3d if t.cachable]
            self.non_cachable_trafos = [t for t in transforms_3d if not t.cachable]
        else:
            transformations = getattr(dataset_any, "transformations", None)
            if transformations is not None:
                self.cachable_trafos = [t for t in transformations if t.cachable]
                self.non_cachable_trafos = [t for t in transformations if not t.cachable]
            else:
                transforms = getattr(dataset_any, "transforms", None)
                if transforms is not None:
                    self.cachable_trafos = [t for t in transforms if (hasattr(t, "cachable") and t.cachable)]
                    self.non_cachable_trafos = [t for t in transforms if not (hasattr(t, "cachable") and t.cachable)]

        if self.cachable_trafos is not None and self.non_cachable_trafos is not None:
            if logger.isEnabledFor(DEBUG_LEVEL_1):
                debug_level_1(f"Cachable trafos: {self.cachable_trafos}")
                debug_level_1(f"Non-cachable trafos: {self.non_cachable_trafos}")

        log_memory_usage("Dataset Initialization")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_time = time.perf_counter()
        if index in (self.shared_hash_map or self.shared_dict):
            _index = index
            if self.shared_hash_map is not None and index in self.shared_hash_map:
                _index = self.shared_hash_map[index]
            logger.debug(f"Retrieving item {_index} from cache.")
            cache = self.shared_dict[_index]
            if self.shared_arrays is not None:
                for key, arr in self.shared_arrays.items():
                    cache[key] = arr[_index]
            cache["index"] = index
            item = copy.deepcopy(cache)
            for name, field in self.dataset.fields.items():
                if not field.cachable:
                    self.dataset.load_field(name, field, item)
        else:
            item = self.dataset.init_item(index)
            cache = item.copy()
            for name, field in self.dataset.fields.items():
                if field.cachable:
                    self.dataset.load_field(name, field, cache)
                else:
                    self.dataset.load_field(name, field, item)

            cache = apply_transforms(cache, self.cachable_trafos)
            item.update(cache)

            _index = index
            if self.shared_hash_map is not None:
                cache.pop("index", None)
                _index = generate_content_hash(cache)
                self.shared_hash_map[index] = _index

            if self.shared_arrays is None:
                if _index not in self.shared_dict:
                    if logger.isEnabledFor(DEBUG):
                        size_in_bytes = 0
                        for v in cache.values():
                            size_in_bytes += v.nbytes if isinstance(v, np.ndarray) else sys.getsizeof(v)
                        size_in_mb = round(size_in_bytes / 1024 / 1024, 2)
                        logger.debug(f"Adding item {_index} to cache (size: {size_in_mb:.2f} MB).")
                    self.shared_dict[_index] = cache
            else:
                for key, value in cache.items():
                    if key in self.shared_arrays:
                        self.shared_arrays[key][_index][:] = value
                non_array_cache = {k: v for k, v in cache.items() if k not in self.shared_arrays}
                self.shared_dict[_index] = non_array_cache

        item = apply_transforms(item, self.non_cachable_trafos)
        logger.debug(f"Loading item {_index} takes {time.perf_counter() - item_time:.4f}s.")
        return item

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        log_memory_usage("Dataset Deletion")


class SharedDataLoader(DataLoader):
    def __init__(self, dataset: ShapeNet, hash_items: bool = False, share_arrays: bool = False, **kwargs):
        self._closed = False
        self._owner_pid = os.getpid()
        self.manager = Manager()
        self.shared_dict: Any = self.manager.dict()
        self.shared_hash_map: Any | None = self.manager.dict() if hash_items else None
        self.shared_arrays: dict[str, np.ndarray] | None = None
        self.shared_memories: dict[str, shared_memory.SharedMemory] | None = None

        if hash_items:
            debug_level_2("Creating shared hash map")

        if share_arrays:
            self.shared_arrays = dict()
            self.shared_memories = dict()
            item = dataset[0]
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    size = (len(dataset), *value.shape)
                    debug_level_2(f"Creating shared memory for {key} with size {size} and dtype {value.dtype}")
                    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(size)) * value.dtype.itemsize)
                    self.shared_memories[key] = shm
                    self.shared_arrays[key] = np.ndarray(size, dtype=value.dtype, buffer=shm.buf)

        super().__init__(
            SharedDataset(
                dataset,
                self.shared_dict,
                self.shared_hash_map,
                self.shared_arrays,
                self.shared_memories,
            ),
            **kwargs,
        )

        # Ensure cleanup even if user forgets to call close()
        atexit.register(self._atexit_close)

        log_memory_usage("DataLoader Initialization")

    def close(self):
        # Idempotent; only the creator process performs unlinking/manager shutdown
        if self._closed:
            return
        self._closed = True

        try:
            if os.getpid() == self._owner_pid:
                if self.shared_memories is not None:
                    for key, sm in list(self.shared_memories.items()):
                        try:
                            debug_level_1(f"Closing/Unlinking shared memory for {key} in SharedDataLoader.close()")
                            sm.close()
                        except Exception as e:
                            debug_level_1(f"SharedMemory.close() failed for {key}: {e}")
                        try:
                            sm.unlink()
                        except FileNotFoundError:
                            pass
                        except Exception as e:
                            debug_level_1(f"SharedMemory.unlink() failed for {key}: {e}")
                try:
                    debug_level_1("Shutting down multiprocessing manager in SharedDataLoader.close()")
                    self.manager.shutdown()
                except Exception as e:
                    debug_level_1(f"Manager shutdown failed or already closed: {e}")
        finally:
            log_memory_usage("DataLoader Close")

    def _atexit_close(self):
        # Best-effort cleanup at interpreter exit
        try:
            self.close()
        except Exception as e:
            debug_level_1(f"Atexit close encountered an error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # Make __del__ a thin wrapper to the idempotent close()
        try:
            self.close()
        except Exception:
            # Avoid noisy errors in interpreter shutdown
            pass
