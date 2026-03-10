"""Compose rendered frames into horizontal strip images.

Pairs with render_generation_process.sh which calls bproc-pubvis per frame.
This script takes a directory of pre-rendered PNGs and composes them into strips.

Usage:
    python render_generation_process.py <input_dir> [--labels] [--show] [--unconditional]
"""

import re
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_rgba(path: Path) -> Image.Image:
    """Load an image into memory and close the underlying file handle."""
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def compose_strip(
    images: list[Image.Image],
    labels: list[str] | None = None,
    padding: int = 4,
    label_height: int = 30,
    bg_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    """Compose images into a horizontal strip with optional labels."""
    if not images:
        raise ValueError("No images to compose")

    w, h = images[0].size
    n = len(images)

    total_w = n * w + (n - 1) * padding
    total_h = h + (label_height if labels else 0)

    strip = Image.new("RGBA", (total_w, total_h), bg_color)

    for i, img in enumerate(images):
        x = i * (w + padding)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        strip.paste(img, (x, 0))

    if labels:
        draw = ImageDraw.Draw(strip)
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 16)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except OSError:
                font = ImageFont.load_default()

        for i, label in enumerate(labels):
            x = i * (w + padding) + w // 2
            y = h + 4
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x - text_w // 2, y), label, fill=(0, 0, 0, 255), font=font)

    return strip


def label_from_stem(stem: str) -> str:
    """Extract a human-readable label from a filename stem."""
    m = re.match(r"^step_(\d+)(?:_.+)?$", stem)
    if m:
        return f"t={m.group(1)}"
    m = re.match(r"^token_(\d+)(?:_.+)?$", stem)
    if m:
        return f"n={m.group(1)}"
    return stem


def extract_suffix(stem: str) -> str:
    """Extract optional view/name suffix from rendered stem."""
    m = re.match(r"^(?:step|token)_\d+_(.+)$", stem)
    if m:
        return m.group(1)
    if stem.startswith("input_"):
        return stem[len("input_"):]
    if stem.startswith("gt_"):
        return stem[len("gt_"):]
    return ""


def collect_timeline(render_dir: Path, suffix: str) -> list[Path]:
    """Collect timeline frames for one modality suffix in numeric order."""
    if suffix:
        step_re = re.compile(rf"^step_(\d+)_{re.escape(suffix)}\.png$")
        token_re = re.compile(rf"^token_(\d+)_{re.escape(suffix)}\.png$")
    else:
        # Important: only unsuffixed shading frames (step_12.png), not step_12_depth.png.
        step_re = re.compile(r"^step_(\d+)\.png$")
        token_re = re.compile(r"^token_(\d+)\.png$")

    steps: list[tuple[int, Path]] = []
    tokens: list[tuple[int, Path]] = []
    for p in render_dir.glob("*.png"):
        m = step_re.match(p.name)
        if m:
            steps.append((int(m.group(1)), p))
            continue
        m = token_re.match(p.name)
        if m:
            tokens.append((int(m.group(1)), p))

    if steps:
        return [p for _, p in sorted(steps, key=lambda x: x[0])]
    if tokens:
        return [p for _, p in sorted(tokens, key=lambda x: x[0])]
    return []


def save_mp4(frames: list[Image.Image], path: Path, fps: float) -> bool:
    """Save RGB frames as MP4 using ffmpeg."""
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found; skipping MP4 export.")
        return False

    with tempfile.TemporaryDirectory(prefix="render_generation_process_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i, frame in enumerate(frames):
            frame.save(tmp_path / f"frame_{i:05d}.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            str(tmp_path / "frame_%05d.png"),
            "-vf",
            "format=yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or "").strip().splitlines()
            tail = err[-5:] if err else ["unknown ffmpeg error"]
            print(f"Warning: ffmpeg failed for {path}:")
            for line in tail:
                print(f"  {line}")
            return False
    return True


def main():
    parser = ArgumentParser(description="Compose rendered frames into strip images.")
    parser.add_argument("input_dir", type=Path,
                        help="Directory with rendered PNGs (output of render_generation_process.sh).")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--method", choices=["diffusion", "ar", "both"], default="both")
    parser.add_argument("--labels", action="store_true", help="Add step labels below frames.")
    parser.add_argument("--unconditional", action="store_true", help="Omit input image panels.")
    parser.add_argument("--gif-duration-ms", type=int, default=120, help="Duration per GIF frame in milliseconds.")
    parser.add_argument("--gif-loop", type=int, default=0, help="GIF loop count (0 = infinite).")
    parser.add_argument(
        "--gif-bg-color",
        nargs=3,
        type=int,
        default=[255, 255, 255],
        metavar=("R", "G", "B"),
        help="Background color used to flatten RGBA PNGs before GIF encoding (default: 255 255 255).",
    )
    parser.add_argument(
        "--animation-format",
        choices=["gif", "mp4", "both"],
        default="gif",
        help="Animation output format for generation process timelines.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=float,
        default=None,
        help="MP4 framerate. Default derives from --gif-duration-ms.",
    )
    parser.add_argument("--implicit", action="store_true",
                        help="Look for PNGs in implicit/ subdirectory (produced by vis.implicit_render=true).")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find object directories: input_dir/<category>/<obj_name>/
    obj_dirs: list[Path] = []
    for cat_dir in sorted(args.input_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            obj_dirs.append(obj_dir)

    if not obj_dirs:
        if (args.input_dir / "diffusion").exists() or (args.input_dir / "ar").exists():
            obj_dirs = [args.input_dir]

    if not obj_dirs:
        print(f"No object directories found in {args.input_dir}")
        return

    methods: list[str] = []
    if args.method in ("diffusion", "both"):
        methods.append("diffusion")
    if args.method in ("ar", "both"):
        methods.append("ar")

    strips_produced = 0

    for obj_dir in obj_dirs:
        rel = obj_dir.relative_to(args.input_dir) if obj_dir != args.input_dir else Path(".")
        obj_out = output_dir / rel
        obj_out.mkdir(parents=True, exist_ok=True)

        for method in methods:
            render_dir = obj_dir / method
            if args.implicit:
                render_dir = render_dir / "implicit"
            if not render_dir.exists():
                continue

            suffixes: set[str] = {""}
            for png in render_dir.glob("*.png"):
                suffix = extract_suffix(png.stem)
                if suffix:
                    suffixes.add(suffix)

            for suffix in sorted(suffixes):
                images: list[Image.Image] = []
                labels_list: list[str] = []

                timeline = collect_timeline(render_dir, suffix)

                if not args.unconditional:
                    input_name = f"input_{suffix}.png" if suffix else "input.png"
                    input_png = render_dir / input_name
                    if input_png.exists():
                        images.append(load_rgba(input_png))
                        labels_list.append("Input")

                for png in timeline:
                    images.append(load_rgba(png))
                    labels_list.append(label_from_stem(png.stem))

                gt_name = f"gt_{suffix}.png" if suffix else "gt.png"
                gt_png = render_dir / gt_name
                if gt_png.exists():
                    images.append(load_rgba(gt_png))
                    labels_list.append("GT")

                if not images:
                    continue

                strip = compose_strip(images, labels=labels_list if args.labels else None)
                strip_name = f"{method}_strip_{suffix}.png" if suffix else f"{method}_strip.png"
                strip_path = obj_out / strip_name
                strip.save(strip_path)
                strips_produced += 1
                print(f"Saved: {strip_path}")

                if timeline:
                    bg_rgba = (*tuple(args.gif_bg_color), 255)
                    gif_frames: list[Image.Image] = []
                    for p in timeline:
                        frame_rgba = load_rgba(p)
                        bg = Image.new("RGBA", frame_rgba.size, bg_rgba)
                        gif_frames.append(Image.alpha_composite(bg, frame_rgba).convert("RGB"))

                    if args.animation_format in ("gif", "both"):
                        gif_name = f"{method}_process_{suffix}.gif" if suffix else f"{method}_process.gif"
                        gif_path = obj_out / gif_name
                        gif_frames[0].save(
                            gif_path,
                            save_all=True,
                            append_images=gif_frames[1:],
                            duration=args.gif_duration_ms,
                            loop=args.gif_loop,
                            disposal=2,
                        )
                        print(f"Saved: {gif_path}")

                    if args.animation_format in ("mp4", "both"):
                        mp4_name = f"{method}_process_{suffix}.mp4" if suffix else f"{method}_process.mp4"
                        mp4_path = obj_out / mp4_name
                        fps = args.mp4_fps if args.mp4_fps is not None else (1000.0 / max(args.gif_duration_ms, 1))
                        if save_mp4(gif_frames, mp4_path, fps):
                            print(f"Saved: {mp4_path}")

                if args.show:
                    strip.show()

        # Combined comparison: diffusion on top, AR on bottom.
        suffixes: set[str] = {""}
        for p in obj_out.glob("diffusion_strip*.png"):
            if p.stem == "diffusion_strip":
                suffixes.add("")
            elif p.stem.startswith("diffusion_strip_"):
                suffixes.add(p.stem[len("diffusion_strip_"):])

        for suffix in sorted(suffixes):
            diff_name = f"diffusion_strip_{suffix}.png" if suffix else "diffusion_strip.png"
            ar_name = f"ar_strip_{suffix}.png" if suffix else "ar_strip.png"
            diff_path = obj_out / diff_name
            ar_path = obj_out / ar_name
            if not (diff_path.exists() and ar_path.exists()):
                continue
            diff_img = load_rgba(diff_path)
            ar_img = load_rgba(ar_path)
            max_w = max(diff_img.width, ar_img.width)
            gap = 8
            combined = Image.new(
                "RGBA",
                (max_w, diff_img.height + gap + ar_img.height),
                (255, 255, 255, 255),
            )
            combined.paste(diff_img, (0, 0))
            combined.paste(ar_img, (0, diff_img.height + gap))
            cmp_name = f"comparison_{suffix}.png" if suffix else "comparison.png"
            cmp_path = obj_out / cmp_name
            combined.save(cmp_path)
            print(f"Saved: {cmp_path}")

    if strips_produced == 0:
        print(f"Warning: no strips produced from {args.input_dir}")
        if args.implicit:
            print("  (--implicit set: expected PNGs in <method>/implicit/ subdirectories)")


if __name__ == "__main__":
    main()
