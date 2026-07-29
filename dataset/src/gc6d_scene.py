from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from .bop_scene import BOPSceneEval, _BOPSample

_GC6D_CAMERAS = ("d415", "d435", "azure_kinect", "zivid")
_GC6D_CAMERA_OFFSETS = {"d415": 1, "d435": 2, "azure_kinect": 3, "zivid": 4}
_GC6D_SPLITS = ("cross_object_train", "cross_object_test", "intra_object_train", "intra_object_test")
_GC6D_SPLIT_FILES = {
    "cross_object_train": "grasp_train_scene_ids.json",
    "cross_object_test": "grasp_test_scene_ids.json",
    "intra_object_train": "ycbv_train_scene_ids.json",
    "intra_object_test": "ycbv_test_scene_ids.json",
}


class GC6DSceneEval(BOPSceneEval):
    """GraspClutter6D scene-level evaluation loader.

    GraspClutter6D ships in BOP-compatible format but differs from the standard
    BOP layout in two ways:

    1. Scenes live directly under ``<root>/scenes/<scene_id>/`` with no
       ``<name>/<split>`` wrapper. All 1,000 scenes are in one directory; the
       split is a logical JSON list, not a physical subdirectory.
    2. Each scene's 52 frames interleave 4 cameras across 13 viewpoints. Frame
       ID ``img_num`` maps to camera via ``img_num % 4``: 1→D415, 2→D435,
       3→Azure Kinect, 0→Zivid (per the official ``graspclutter6dAPI`` loader).

    Visible masks use the standard BOP ``mask_visib/`` directory; ``mask/``
    contains the amodal masks. The released scene archives do not contain a
    ``visible_mask/`` directory. Meshes in ``models_eval/`` are in millimeters,
    so keep the default ``mesh_scale=0.001``.
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal[
            "cross_object_train",
            "cross_object_test",
            "intra_object_train",
            "intra_object_test",
        ],
        camera: Literal["d415", "d435", "azure_kinect", "zivid"] | None = None,
        scene_ids: Iterable[int] | None = None,
        name: str = "graspclutter6d",
        **kwargs: Any,
    ) -> None:
        if split not in _GC6D_SPLITS:
            raise ValueError(f"Unknown GraspClutter6D split: {split!r}. Expected one of {_GC6D_SPLITS}.")
        if camera is not None and camera not in _GC6D_CAMERAS:
            raise ValueError(f"Unknown GraspClutter6D camera: {camera!r}. Expected one of {_GC6D_CAMERAS}.")

        root = Path(root)
        split_scene_ids = self._load_split_scene_ids(root, split)
        if scene_ids is not None:
            requested = set(scene_ids)
            missing = requested - set(split_scene_ids)
            if missing:
                raise ValueError(f"Requested scene_ids {sorted(missing)} are not in GraspClutter6D split {split!r}.")
            split_scene_ids = [s for s in split_scene_ids if s in requested]

        if kwargs.get("mesh_dir") is None:
            kwargs["mesh_dir"] = root / "models_eval"
        kwargs.setdefault("mask_dir", "mask_visib")
        kwargs.setdefault("target_filename", None)

        self.gc6d_split = split
        self.gc6d_camera = camera
        self.gc6d_camera_offset = _GC6D_CAMERA_OFFSETS.get(camera) if camera is not None else None

        super().__init__(
            root=root,
            name=name,
            split="scenes",
            scene_ids=split_scene_ids,
            **kwargs,
        )
        self.split = split
        self.camera = camera

    @staticmethod
    def _load_split_scene_ids(root: Path, split: str) -> list[int]:
        filename = _GC6D_SPLIT_FILES[split]
        path = root / "split_info" / filename
        if not path.exists():
            raise FileNotFoundError(
                f"GraspClutter6D split file not found: {path}. Download split_info.7z from the dataset repo."
            )
        with path.open("r", encoding="utf-8") as f:
            ids = json.load(f)
        return [int(x) for x in ids]

    def _find_scene_dirs(self) -> list[Path]:
        scenes_dir = self.root / "scenes"
        if not scenes_dir.exists():
            raise FileNotFoundError(f"Could not locate GraspClutter6D scenes directory {scenes_dir}.")
        if self.scene_ids is None:
            raise ValueError("GraspClutter6D requires an explicit scene split.")
        scene_names = {f"{scene_id:06d}" for scene_id in self.scene_ids}
        scene_dirs = sorted(
            p for p in scenes_dir.iterdir() if p.is_dir() and p.name in scene_names and (p / self.depth_dir).exists()
        )
        if not scene_dirs:
            raise FileNotFoundError(
                f"No GraspClutter6D scene directories found in {scenes_dir} for the requested split."
            )
        return scene_dirs

    def _enumerate_samples(self) -> list[_BOPSample]:
        samples: list[_BOPSample] = []
        for scene_dir in self.scene_dirs:
            ann_ids = sorted(p.stem for p in (scene_dir / self.depth_dir).glob("*.png"))
            if self.gc6d_camera_offset is not None:
                ann_ids = [a for a in ann_ids if self._frame_matches_camera(int(a))]
            if self.one_view_per_scene and ann_ids:
                samples.append(_BOPSample(scene_dir=scene_dir, ann_id=ann_ids[len(ann_ids) // 2]))
            else:
                samples.extend(_BOPSample(scene_dir=scene_dir, ann_id=ann_id) for ann_id in ann_ids)
        if not samples:
            raise FileNotFoundError(
                f"No depth frames found for GraspClutter6D split {self.gc6d_split!r}"
                f"{' camera=' + self.gc6d_camera if self.gc6d_camera else ''}."
            )
        return samples

    def _frame_matches_camera(self, img_num: int) -> bool:
        return self.gc6d_camera_offset is None or img_num % 4 == self.gc6d_camera_offset % 4


__all__ = ["GC6DSceneEval"]
