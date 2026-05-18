# Workflow status

This repository contains research code with several levels of support.

## Supported for public sanity checks

- Hydra config composition for every runner-backed paper recipe.
- CLI rendering of reproduction commands through `scripts/reproduce_experiment.py`.
- CPU-safe unit tests for utilities, datasets, evaluation helpers, and public APIs.
- Source-only linting and type checking through the validation scripts.

## Requires local data or checkpoints

- Paper reproduction beyond command/config sanity.
- `gen_eval`, `mesh_eval`, and BOP scene evaluation on real datasets.
- Any recipe that accepts `weights`, `vae_weights`, `occ_weights`, or `inst_weights`.

## Requires compiled or GPU dependencies

- CUDA extension tests for Chamfer/EMD/fusion/MISE-style components.
- tiny-cuda-nn backends.
- GPU training and full evaluation runs.
- Renderer tests on systems without stable headless rendering support.

## Experimental

- Long-running sweeps and ablation scripts.
- Some visualization and rendering workflows.
- Optional third-party backends that are not installed by the base package.

When adding public workflows, prefer a CPU-safe sanity test first and keep machine-specific setup in local, untracked configuration.
