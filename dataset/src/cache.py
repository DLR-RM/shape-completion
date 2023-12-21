import copy
from typing import Any, Dict

from torch.utils.data import Dataset

from utils import setup_logger
from .transforms import apply_transform


logger = setup_logger(__name__)


class CachedDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 verbose: bool = False):
        self.dataset = dataset
        self.cache = dict()
        self.transform = None if not hasattr(dataset, "augmentation") else dataset.augmentation
        self.verbose = verbose

    def clear_fields_caches(self):
        if hasattr(self.dataset, "fields") and isinstance(self.dataset.fields, dict):
            for name, field in self.dataset.fields.items():
                logger.debug(f"Clearing cache of field {name}")
                field.clear_cache()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index in self.cache:
            data = copy.deepcopy(self.cache[index])
            data = apply_transform(data, self.transform, self.verbose)
        else:
            item = self.dataset[index]
            self.cache[index] = item
            data = copy.deepcopy(item)
        return data

    def __len__(self):
        return len(self.dataset)
