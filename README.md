# Shape Completion
This is the official repository for the following publications:

| Title | Authors | Venue |
|---|---|---|
| [**Shape Completion with Prediction of Uncertain Regions**](https://arxiv.org/abs/2308.00377) | Matthias Humt, Dominik Winkelbauer, Ulrich Hillenbrand | IROS 2023 |

Shape completion only:

| Title | Authors | Venue |
|---|---|---|
| [**Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand**](https://dlr-alr.github.io/grasping/_pages/humanoids23.html) | Matthias Humt, Dominik Winkelbauer, Ulrich Hillenbrand, Berthold Bäumle | Humanoids 2023 |

## Code
Almost ready for release.

## Datasets
### 1. [Shape Completion with Prediction of Uncertain Regions](https://zenodo.org/uploads/10284230)

**Overview**

This [dataset](https://zenodo.org/uploads/10284230) contains rendered depth images, watertight meshes and occupancy as well as uncertain region labels for the `mugs` category (`03797390`) of the [`ShapeNetCore.v1`](https://shapenet.org) dataset.

It further contains optimized, watertight meshes and occupancy as well as uncertain region labels for the mugs found in the `HB`, `LM`, `TYOL` and `YCBV` datasets from the [`BOP`](https://bop.felk.cvut.cz/datasets) challenge.

**Structure**

After downloading the `shapenet.tar.gz` and `bop.tar.gz` files as well as the original datasets, simply unpack and move the content to the corresponding directories. The structure should be as follows:
```bash
.
├── shapenet
│   └── 03797390
│       ├── 1038e4eac0e18dcce02ae6d2a21d494a
|       |   ├── blenderproc
│       │   ├── mesh
│       │   ├── model.binvox
│       │   ├── pointcloud.npz
│       │   ├── points.npz
│       │   └── samples
│       ├── 10c2b3eac377b9084b3c42e318f3affc
│       ├── 10f6e09036350e92b3f21f1137c3c347
|       ├── ...
|       ├── train.lst
|       ├── val.lst
|       └── test.lst
└── bop
    ├── hb
    ├── lm
    ├── tyol
    └── ycbv
```

**Code**

Python code for loading of the dataset is provided in [`datasets/shapenet.py`]() and [`datasets/bop.py`]() as well as [`datasets/fields.py`]().

**Citation**

If you find the provided dataset useful in your research, please use the following BibTex entry to cite the corresponding research paper:

```bibtex
@misc{humt2023shape,
      title={Shape Completion with Prediction of Uncertain Regions}, 
      author={Matthias Humt and Dominik Winkelbauer and Ulrich Hillenbrand},
      year={2023},
      eprint={2308.00377},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
