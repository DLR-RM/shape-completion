from __future__ import annotations

import argparse
import shlex
import string
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

SUITES = ("iros2023", "humanoids2023", "3dv2026", "inst3d2026")
STAGES = ("preprocess", "train", "eval")


@dataclass(frozen=True)
class Recipe:
    suite: str
    name: str
    stage: str
    description: str
    command: tuple[str, ...]
    use_runner: bool = True

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.name}"

    @property
    def required_vars(self) -> tuple[str, ...]:
        names: set[str] = set()
        formatter = string.Formatter()
        for token in self.command:
            for _, field_name, _, _ in formatter.parse(token):
                if field_name:
                    names.add(field_name)
        return tuple(sorted(names))

    def render_args(self, values: dict[str, str], *, runner: str = "./scripts/run.sh") -> list[str]:
        missing = [name for name in self.required_vars if name not in values]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"{self.key} requires: {missing_list}")

        args = [token.format(**values) for token in self.command]
        if self.use_runner:
            args = [runner, *args]
        return args

    def render_line(self, values: dict[str, str], *, runner: str = "./scripts/run.sh") -> str:
        return shlex.join(self.render_args(values, runner=runner))


def _r(
    suite: str,
    name: str,
    stage: str,
    description: str,
    *command: str,
    use_runner: bool = True,
) -> Recipe:
    return Recipe(suite=suite, name=name, stage=stage, description=description, command=command, use_runner=use_runner)


RECIPES: tuple[Recipe, ...] = (
    _r(
        "iros2023",
        "watertight-meshes",
        "preprocess",
        "Generate watertight meshes before point sampling or depth rendering.",
        "python3",
        "-m",
        "process.scripts.make_watertight",
        "{mesh_root}",
        "--out_dir",
        "{watertight_root}",
        "--in_format",
        ".obj",
        "--out_format",
        ".off",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "iros2023",
        "sample-points",
        "preprocess",
        "Sample occupancy and surface points from watertight meshes.",
        "python3",
        "-m",
        "process.scripts.process_mesh",
        "{watertight_root}",
        "sample",
        "--out_dir",
        "{samples_root}",
        "--in_format",
        ".off",
        "--num_points",
        "100000",
        "--normalize",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "iros2023",
        "uncertain-labels",
        "preprocess",
        "Regenerate uncertain-region labels for ShapeNet-style depth data.",
        "python3",
        "-m",
        "process.scripts.find_uncertain_regions",
        "-cn",
        "shapenet_uncertain",
        use_runner=False,
    ),
    _r(
        "iros2023",
        "mugs-main-train",
        "train",
        "Train the ConvONet uncertain-region model.",
        "train",
        "-cn",
        "mugs_paper",
    ),
    _r("iros2023", "mugs-dropout-train", "train", "Train the MCDropout baseline.", "train", "-cn", "mugs_dropout"),
    _r("iros2023", "mugs-pssnet-train", "train", "Train the PSSNet baseline.", "train", "-cn", "mugs_pssnet"),
    _r("iros2023", "mugs-realnvp-train", "train", "Train the RealNVP baseline.", "train", "-cn", "mugs_realnvp"),
    _r(
        "iros2023",
        "mugs-main-eval",
        "eval",
        "Evaluate a trained uncertain-region model.",
        "gen_eval",
        "-cn",
        "mugs_paper",
        "model.weights={weights}",
    ),
    _r(
        "humanoids2023",
        "watertight-meshes",
        "preprocess",
        "Generate watertight meshes for Automatica-style objects.",
        "python3",
        "-m",
        "process.scripts.make_watertight",
        "{mesh_root}",
        "--out_dir",
        "{watertight_root}",
        "--in_format",
        ".obj",
        "--out_format",
        ".off",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "humanoids2023",
        "render-kinect",
        "preprocess",
        "Render one Kinect-style depth image from a mesh.",
        "python3",
        "-m",
        "process.scripts.render_kinect",
        "{mesh}",
        "--out_dir",
        "{depth_root}",
        "--n_views",
        "1",
        use_runner=False,
    ),
    _r(
        "humanoids2023",
        "render-kinect-parallel",
        "preprocess",
        "Render Kinect-style depth images for a directory of meshes.",
        "python3",
        "-m",
        "process.scripts.render_kinect_parallel",
        "{watertight_root}",
        "--out_dir",
        "{depth_root}",
        "--in_format",
        ".off",
        "--n_jobs",
        "{n_jobs}",
        "--n_views",
        "100",
        use_runner=False,
    ),
    _r(
        "humanoids2023",
        "stable-poses",
        "preprocess",
        "Generate stable resting poses for grasping datasets.",
        "python3",
        "-m",
        "process.scripts.generate_physics_poses",
        "{watertight_root}",
        "--out_dir",
        "{poses_root}",
        "--in_format",
        ".off",
        "--num_poses",
        "100",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "humanoids2023",
        "vqdif-train",
        "train",
        "Train the VQDIF shape-completion model.",
        "train",
        "-cn",
        "humanoids_2023",
    ),
    _r(
        "humanoids2023",
        "point-transformer-train",
        "train",
        "Train the PointTransformer comparison model.",
        "train",
        "-cn",
        "automatica_2023_kinect_gpt2",
    ),
    _r(
        "humanoids2023",
        "shapeformer-train",
        "train",
        "Train the ShapeFormer comparison model.",
        "train",
        "-cn",
        "shapenet_v1_depth_shapeformer",
    ),
    _r(
        "humanoids2023",
        "vqdif-eval",
        "eval",
        "Evaluate a trained VQDIF model.",
        "gen_eval",
        "-cn",
        "humanoids_2023",
        "model.weights={weights}",
    ),
    _r(
        "3dv2026",
        "watertight-meshes",
        "preprocess",
        "Generate watertight ShapeNet meshes for latent generative experiments.",
        "python3",
        "-m",
        "process.scripts.make_watertight",
        "{mesh_root}",
        "--out_dir",
        "{watertight_root}",
        "--in_format",
        ".obj",
        "--out_format",
        ".off",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "3dv2026",
        "sample-points",
        "preprocess",
        "Sample point-cloud and occupancy training data from watertight meshes.",
        "python3",
        "-m",
        "process.scripts.process_mesh",
        "{watertight_root}",
        "sample",
        "--out_dir",
        "{samples_root}",
        "--in_format",
        ".off",
        "--num_points",
        "100000",
        "--normalize",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "3dv2026",
        "render-kinect-parallel",
        "preprocess",
        "Render Kinect-style partial observations for conditional runs.",
        "python3",
        "-m",
        "process.scripts.render_kinect_parallel",
        "{watertight_root}",
        "--out_dir",
        "{depth_root}",
        "--in_format",
        ".off",
        "--n_jobs",
        "{n_jobs}",
        "--n_views",
        "100",
        use_runner=False,
    ),
    _r(
        "3dv2026",
        "pack-hdf5",
        "preprocess",
        "Pack processed object directories into HDF5 files.",
        "python3",
        "-m",
        "dataset.scripts.process_dataset",
        "{samples_root}",
        "pack",
        "--out_dir",
        "{hdf5_root}",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "3dv2026",
        "vae-train",
        "train",
        "Train the Shape3D2VecSet VAE stage.",
        "train",
        "-cn",
        "cvpr_2025_vae",
        "train.batch_size=128",
        "train.epochs=800",
        "train.lr=7.8125e-07",
    ),
    _r(
        "3dv2026",
        "vqvae-train",
        "train",
        "Train the VQ-VAE stage used by AR and VQ-latent LDM runs.",
        "train",
        "-cn",
        "cvpr_2025_vae",
        "model.arch=3dshape2vecset_vqvae",
        "+n_code=16384",
        "train.batch_size=128",
        "train.epochs=800",
        "train.lr=7.8125e-07",
    ),
    _r(
        "3dv2026",
        "ldm-uncond-train",
        "train",
        "Train the unconditional latent diffusion model on VAE latents.",
        "train",
        "-cn",
        "cvpr_2025",
        "model.average=ema",
        "model.ema_decay=0.999",
        "train.batch_size=64",
        "train.epochs=2000",
        "train.lr=3.90625e-07",
        "+vae_weights={vae_weights}",
    ),
    _r(
        "3dv2026",
        "ar-uncond-train",
        "train",
        "Train the unconditional latent autoregressive model on VQ-VAE latents.",
        "train",
        "-cn",
        "cvpr_2025",
        "model.arch=larm",
        "model.compile=False",
        "model.average=ema",
        "model.ema_decay=0.999",
        "++vae_arch=3dshape2vecset_vqvae",
        "+vae_weights={vae_weights}",
        "+n_code=16384",
        "train.batch_size=64",
        "train.epochs=2000",
        "train.lr=3.90625e-07",
    ),
    _r(
        "3dv2026",
        "ldm-kinect-train",
        "train",
        "Train the Kinect-conditioned latent diffusion model.",
        "train",
        "-cn",
        "cvpr_2025_cond",
        "data.frame=net",
        "norm.reference=inputs.depth",
        "norm.center=True",
        "norm.scale=True",
        "train.batch_size=64",
        "train.epochs=2000",
        "train.lr=3.90625e-07",
        "+vae_weights={vae_weights}",
    ),
    _r(
        "3dv2026",
        "ar-kinect-train",
        "train",
        "Train the Kinect-conditioned autoregressive model.",
        "train",
        "-cn",
        "cvpr_2025_cond",
        "model.arch=larm",
        "model.compile=False",
        "model.average=ema",
        "model.ema_decay=0.999",
        "data.frame=net",
        "norm.reference=inputs.depth",
        "norm.center=True",
        "norm.scale=True",
        "++vae_arch=3dshape2vecset_vqvae",
        "+vae_weights={vae_weights}",
        "train.batch_size=64",
        "train.epochs=2000",
        "train.lr=3.90625e-07",
    ),
    _r(
        "3dv2026",
        "ldm-uncond-eval",
        "eval",
        "Evaluate an unconditional LDM checkpoint with the matching VAE.",
        "gen_eval",
        "-cn",
        "cvpr_2025",
        "model.weights={weights}",
        "+vae_weights={vae_weights}",
        "train.batch_size=64",
        "train.epochs=2000",
        "train.lr=3.90625e-07",
        "norm.center=True",
        "norm.scale=True",
    ),
    _r(
        "3dv2026",
        "ar-uncond-eval",
        "eval",
        "Evaluate an unconditional AR checkpoint with the matching VQ-VAE.",
        "gen_eval",
        "-cn",
        "cvpr_2025",
        "model.weights={weights}",
        "model.arch=larm",
        "model.compile=False",
        "model.average=ema",
        "model.ema_decay=0.999",
        "++vae_arch=3dshape2vecset_vqvae",
        "+vae_weights={vae_weights}",
        "+n_code=16384",
        "train.batch_size=64",
        "train.epochs=2000",
        "train.lr=3.90625e-07",
        "norm.center=True",
        "norm.scale=True",
    ),
    _r(
        "inst3d2026",
        "render-tabletop-packed",
        "preprocess",
        "Render packed tabletop scenes with the public tabletop rendering wrapper.",
        "./scripts/render_tabletop.sh",
        "train",
        "{num_scenes}",
        "{num_views}",
        "{shard}",
        "{tabletop_dataset}",
        "packed",
        use_runner=False,
    ),
    _r(
        "inst3d2026",
        "render-tabletop-pile",
        "preprocess",
        "Render pile tabletop scenes with the public tabletop rendering wrapper.",
        "./scripts/render_tabletop.sh",
        "train",
        "{num_scenes}",
        "{num_views}",
        "{shard}",
        "{tabletop_dataset}",
        "pile",
        use_runner=False,
    ),
    _r(
        "inst3d2026",
        "add-kinect-sim",
        "preprocess",
        "Add the kinect_sim modality to BlenderProc HDF5 scene files.",
        "python3",
        "-m",
        "process.scripts.add_kinect_sim",
        "--input-dir",
        "{hdf5_root}",
        "--shapenet-dir",
        "{shapenet_root}",
        "--noise",
        "perlin",
        use_runner=False,
    ),
    _r(
        "inst3d2026",
        "pack-hdf5",
        "preprocess",
        "Pack rendered scene directories into HDF5 files.",
        "python3",
        "-m",
        "dataset.scripts.process_dataset",
        "{hdf5_root}",
        "pack",
        "--out_dir",
        "{packed_root}",
        "--n_jobs",
        "{n_jobs}",
        use_runner=False,
    ),
    _r(
        "inst3d2026",
        "joint-train",
        "train",
        "Train the joint instance-segmentation and completion model.",
        "train",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "inputs.num_points=100000",
        "inputs.type=kinect",
        "aug.noise=null",
        "+multitask=head",
        "+inputs_weight=0.2",
        "+mlp_heads=True",
    ),
    _r(
        "inst3d2026",
        "pipeline-instseg-train",
        "train",
        "Train the instance-segmentation stage for the pipeline baseline.",
        "train",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "inputs.num_points=100000",
        "inputs.type=kinect",
        "aug.noise=null",
    ),
    _r(
        "inst3d2026",
        "joint-pile-ft",
        "train",
        "Fine-tune the joint model on tabletop pile data.",
        "train",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "model.weights={weights}",
        "inputs.num_points=100000",
        "inputs.type=kinect_sim",
        "data.train_ds=[tabletop_pile]",
        "test.no_aug=True",
        "aug.noise=null",
        "train.lr=1e-4",
        "train.epochs=10",
        "train.scheduler=LinearWarmupCosineAnnealingLR",
        "train.min_lr=1e-6",
        "+multitask=head",
        "+inputs_weight=0.2",
        "+mlp_heads=True",
    ),
    _r(
        "inst3d2026",
        "pipeline-pile-eval",
        "eval",
        "Evaluate the two-stage pipeline on tabletop pile data.",
        "evaluate",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "model.arch=inst_pipe",
        "inputs.type=kinect_sim",
        "data.train_ds=[tabletop_pile]",
        "test.no_aug=True",
        "points.subsample=False",
        "+apply_filter=False",
        "+align_to_gt=True",
        "+occ_weights={occ_weights}",
        "+inst_weights={inst_weights}",
    ),
    _r(
        "inst3d2026",
        "bop-joint-eval",
        "eval",
        "Evaluate a camera-frame joint checkpoint on a BOP scene split.",
        "evaluate",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "model.weights={weights}",
        "inputs.type=kinect",
        "inputs.crop=511",
        "data.frame=cam",
        "aug.rotate=False",
        "points.subsample=False",
        "+apply_filter=False",
        "+align_to_gt=True",
        "+single_view=False",
        "+refine_pose=False",
        "data.test_ds=[{bop_split}]",
        "+multitask=head",
        "+inputs_weight=0.2",
        "+mlp_heads=True",
    ),
    _r(
        "inst3d2026",
        "bop-pipeline-eval",
        "eval",
        "Evaluate camera-frame pipeline checkpoints on a BOP scene split.",
        "evaluate",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "model.arch=inst_pipe",
        "inputs.type=kinect",
        "inputs.crop=511",
        "data.frame=cam",
        "aug.rotate=False",
        "points.subsample=False",
        "+apply_filter=False",
        "+align_to_gt=True",
        "+single_view=False",
        "+refine_pose=False",
        "data.test_ds=[{bop_split}]",
        "+occ_weights={occ_weights}",
        "+inst_weights={inst_weights}",
    ),
    _r(
        "inst3d2026",
        "bop-oracle-eval",
        "eval",
        "Evaluate a camera-frame completion checkpoint with ground-truth BOP masks.",
        "evaluate",
        "-cn",
        "tabletop_inst_seg_pcd_3d",
        "model.arch=inst_pipe",
        "inputs.type=kinect",
        "inputs.crop=511",
        "data.frame=cam",
        "aug.rotate=False",
        "points.subsample=False",
        "+apply_filter=False",
        "+align_to_gt=True",
        "+single_view=False",
        "+refine_pose=False",
        "data.test_ds=[{bop_split}]",
        "+occ_weights={occ_weights}",
    ),
)


def iter_recipes(*, suite: str | None = None, stage: str | None = None, name: str | None = None) -> Iterable[Recipe]:
    for recipe in RECIPES:
        if suite is not None and recipe.suite != suite:
            continue
        if stage is not None and recipe.stage != stage:
            continue
        if name is not None and recipe.name != name:
            continue
        yield recipe


def get_recipe(key: str) -> Recipe:
    if ":" not in key:
        matches = list(iter_recipes(name=key))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(recipe.key for recipe in matches)
            raise ValueError(f"recipe name {key!r} is ambiguous; use one of: {choices}")
    for recipe in RECIPES:
        if recipe.key == key:
            return recipe
    raise ValueError(f"unknown recipe: {key}")


def parse_values(items: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--set expects a non-empty key, got {item!r}")
        values[key] = value
    return values


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List, render, or execute public paper reproduction recipes.")
    parser.add_argument("--list", action="store_true", help="List matching recipes and exit.")
    parser.add_argument("--suite", choices=SUITES, help="Limit recipes to one paper or functionality suite.")
    parser.add_argument("--stage", choices=STAGES, help="Limit recipes to one stage.")
    parser.add_argument("--recipe", help="Render or execute one recipe, either NAME or SUITE:NAME.")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Value for recipe placeholders."
    )
    parser.add_argument("--runner", default="./scripts/run.sh", help="Runner used for Hydra entry points.")
    parser.add_argument(
        "--execute", action="store_true", help="Execute commands. Without this flag, commands are printed."
    )
    return parser


def _print_recipe_list(recipes: Sequence[Recipe]) -> None:
    for recipe in recipes:
        required = ", ".join(recipe.required_vars) or "-"
        print(f"{recipe.key}\t{recipe.stage}\trequires: {required}\t{recipe.description}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        values = parse_values(args.set)
        if args.recipe:
            recipes = [get_recipe(args.recipe)]
            if args.suite and recipes[0].suite != args.suite:
                raise ValueError(f"{recipes[0].key} does not belong to suite {args.suite!r}")
            if args.stage and recipes[0].stage != args.stage:
                raise ValueError(f"{recipes[0].key} does not belong to stage {args.stage!r}")
        else:
            recipes = list(iter_recipes(suite=args.suite, stage=args.stage))

        if args.list or not args.execute:
            if args.list:
                _print_recipe_list(recipes)
            else:
                for recipe in recipes:
                    print(recipe.render_line(values, runner=args.runner))
            return 0

        for recipe in recipes:
            subprocess.run(recipe.render_args(values, runner=args.runner), check=True, cwd=repo_root())
    except ValueError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
