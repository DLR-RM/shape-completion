import argparse
import importlib.metadata
import logging
import os
import pathlib
import shutil
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def _uv_available() -> bool:
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_pip_command(use_uv: bool = True) -> list[str]:
    """Return ['uv', 'pip'] if use_uv is True and uv is available, else ['pip']."""
    if use_uv and _uv_available():
        return ["uv", "pip"]
    return ["pip"]


def get_libraries(names: list[str] | None = None) -> list[str]:
    libs = [
        name
        for name in os.listdir(_LIBS_DIR)
        if os.path.isdir(os.path.join(_LIBS_DIR, name)) and "lib" in name and name != "libfusion"
    ]
    libs.extend(["libfusion_cpu", "libfusion_gpu"])
    if names is None or not names:
        return libs
    return [lib for lib in libs if lib.removeprefix("lib") in names]


def _get_pkg_name(lib_name: str) -> str:
    """Map a lib directory name to its installed package name."""
    if "fusion" in lib_name:
        return "pyfusion_cpu" if "cpu" in lib_name else "pyfusion_gpu"
    if "kinect" in lib_name:
        return "kinect_ext"
    return lib_name.removeprefix("lib") + "_ext"


def _is_installed(lib_name: str) -> bool:
    """Check that the package has metadata AND its code files exist on disk.

    Metadata alone (.dist-info) can survive a ``uv sync`` that deletes the
    actual ``.so``/``.py`` files, giving a false positive.
    """
    try:
        dist = importlib.metadata.distribution(_get_pkg_name(lib_name))
    except importlib.metadata.PackageNotFoundError:
        return False
    if not dist.files:
        return False
    site_pkgs = pathlib.Path(dist._path).parent  # type: ignore[attr-defined]
    return any(
        (site_pkgs / f).exists()
        for f in dist.files
        if str(f).endswith((".so", ".pyd", ".py")) and "dist-info" not in str(f)
    )


def install_library(
    lib_name: str,
    upgrade: bool = False,
    force: bool = False,
    verbose: bool = False,
    use_uv: bool = False,
    build_isolation: bool = False,
):
    if not upgrade and not force and _is_installed(lib_name):
        logger.info(
            f"Library `{lib_name}` already installed, skipping. Use 'upgrade' or 'force-reinstall' to overwrite."
        )
        return

    path_to_setup = os.path.join(_LIBS_DIR, lib_name)
    if "fusion" in lib_name:
        path_to_setup = os.path.join(_LIBS_DIR, "libfusion")
        if "cpu" in lib_name:
            path_to_setup = os.path.join(path_to_setup, "cpu")
        else:
            path_to_setup = os.path.join(path_to_setup, "gpu")
    try:
        args = [*get_pip_command(use_uv), "install"]
        if verbose:
            args.append("-v")
        if upgrade:
            args.append("-U")
        if force:
            args.append("--force-reinstall")
        if not build_isolation:
            args.append("--no-build-isolation")
        args.append(path_to_setup)
        subprocess.run(args, check=True)
        logger.debug(f"Library `{lib_name}` installed successfully.")
    except subprocess.CalledProcessError:
        logger.error(f"Installation of `{lib_name}` failed.")


def uninstall_library(lib_name: str, use_uv: bool = False):
    pkg_name = _get_pkg_name(lib_name)
    try:
        subprocess.run([*get_pip_command(use_uv), "uninstall", "-y", pkg_name])
        logger.debug(f"Library `{lib_name}` uninstalled successfully.")
    except subprocess.CalledProcessError:
        logger.error(f"Uninstallation of `{lib_name}` failed.")


def clean_library(lib_name: str, clean_jit: bool = True):
    if clean_jit:
        for d in [
            "/tmp",
            os.path.expanduser("~"),
            os.path.expandvars("$XDG_CACHE_HOME"),
        ]:
            shutil.rmtree(os.path.join(d, "torch_extensions"), ignore_errors=True)

    path_to_setup = os.path.join(_LIBS_DIR, lib_name)
    if "fusion" in lib_name:
        sub = "cpu" if "cpu" in lib_name else "gpu"
        path_to_setup = os.path.join(_LIBS_DIR, "libfusion", sub)
    for d in os.listdir(path_to_setup):
        p = os.path.join(path_to_setup, d)
        if os.path.isdir(p) and (d in ["build", "dist"] or "egg-info" in d):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            if d.endswith(".pyx"):
                cpp = p.replace(".pyx", ".cpp")
                if os.path.isfile(cpp):
                    os.remove(cpp)
            elif clean_jit and (d.endswith(".o") or d.endswith(".so") or "ninja" in d):
                os.remove(p)
    logger.debug(f"Library `{lib_name}` cleaned successfully.")


def main():
    parser = argparse.ArgumentParser(description="Manage libraries")
    parser.add_argument(
        "command",
        choices=["install", "uninstall", "upgrade", "force-reinstall", "clean"],
        type=str,
        help="Action to perform",
    )
    parser.add_argument("names", nargs="*", type=str, help="Name(s) of librarie(s) to manage")
    parser.add_argument(
        "--cuda_archs",
        default="5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX",
        type=str,
        help="Comma-separated list of CUDA architectures to compile for",
    )
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--no-uv", action="store_true", help="Use pip instead of uv")
    parser.add_argument(
        "--build-isolation",
        action="store_true",
        help="Enable build isolation (default: off, reuses venv's torch for faster builds)",
    )
    args = parser.parse_args()

    env_archs = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_archs and env_archs != args.cuda_archs:
        raise ValueError(
            f"CUDA architectures mismatch. TORCH_CUDA_ARCH_LIST={env_archs!r}, --cuda_archs={args.cuda_archs!r}"
        )
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.cuda_archs

    libs = get_libraries([name.removeprefix("lib") for name in args.names])
    if libs:
        name = "library" if len(libs) == 1 else "libraries"
        if args.verbose:
            logger.debug(f"Found {len(libs)} {name}: {', '.join(libs)}")
        else:
            logger.info(f"Found {len(libs)} {name}")

        for lib in libs:
            if args.command == "install":
                install_library(
                    lib,
                    verbose=args.verbose,
                    use_uv=not args.no_uv,
                    build_isolation=args.build_isolation,
                )
            elif args.command == "uninstall":
                uninstall_library(lib, use_uv=not args.no_uv)
            elif args.command == "upgrade":
                install_library(
                    lib,
                    upgrade=True,
                    verbose=args.verbose,
                    use_uv=not args.no_uv,
                    build_isolation=args.build_isolation,
                )
            elif args.command == "force-reinstall":
                install_library(
                    lib,
                    force=True,
                    verbose=args.verbose,
                    use_uv=not args.no_uv,
                    build_isolation=args.build_isolation,
                )
            elif args.command == "clean":
                clean_library(lib)
    else:
        logger.warning("No libraries found.")


if __name__ == "__main__":
    main()
