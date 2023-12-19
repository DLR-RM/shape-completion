# Shape Completion

This is the official repository for the following publications:

| Title | Authors | Venue |
|---|---|---|
| [**Shape Completion with Prediction of Uncertain Regions**](https://arxiv.org/abs/2308.00377) | Matthias Humt, Dominik Winkelbauer, Ulrich Hillenbrand | IROS 2023 |

Shape completion only:

| Title | Authors | Venue |
|---|---|---|
| [**Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand**](https://dlr-alr.github.io/grasping/_pages/humanoids23.html) | Matthias Humt, Dominik Winkelbauer, Ulrich Hillenbrand, Berhold Bäumle | Humanoids 2023 |

## Code
Almost ready for release.

## Datasets
### 1. [**Shape Completion with Prediction of Uncertain Regions**](https://zenodo.org/uploads/10284230)
This dataset contains rendered depth images, wateright meshes and occupancy as well as uncertain region labels for the `mugs` category (`03797390`) of the [`ShapeNetCore.v1`](https://shapenet.org/) dataset.

It further contains optimized, watertight meshes and occupancy as well as uncertain region labels for the mugs found in the `HB`, `LM`, `TYOL` and `YCBV` datasets from the [`BOP`](https://bop.felk.cvut.cz/datasets/) challenge.

After downloading the original datasets, simply unpack the provided `shapenet.tar.gz` and `bop.tar.gz` into their respective directories.
