import argparse
import logging
import os
import subprocess
import shutil
from typing import List, Optional


logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_libraries(names: Optional[List[str]] = None):
    libs = [name for name in os.listdir('libs') if os.path.isdir(os.path.join('libs', name)) and 'lib' in name]
    if 'libkinect' in libs or 'kinect' in names or 'libkinect' in names:
        logger.warning('Kinect library is not supported yet.')
        libs.remove('libkinect')
    if names is None or not names:
        return libs
    return [lib for lib in libs if lib.removeprefix('lib') in names]


def install_library(lib_name):
    path_to_setup = os.path.join('libs', lib_name)
    if 'fusion' in lib_name:
        path_to_setup = os.path.join(path_to_setup, 'gpu')
    try:
        subprocess.run(['pip', 'install', path_to_setup], check=True)
    except subprocess.CalledProcessError:
        logger.error(f'Installation of `{lib_name}` failed.')


def uninstall_library(lib_name):
    pkg_name = lib_name.removeprefix('lib')
    if 'pointnet' in pkg_name:
        pkg_name = 'pointnet2_ops'
    elif 'fusion' in pkg_name:
        pkg_name = 'pyfusion_gpu'
    elif 'simplify' in pkg_name:
        pkg_name = 'simplify_mesh'
    subprocess.run(['pip', 'uninstall', '-y', pkg_name])


def update_library(lib_name):
    path_to_setup = os.path.join('libs', lib_name)
    if 'fusion' in lib_name:
        path_to_setup = os.path.join(path_to_setup, 'gpu')
    subprocess.run(['pip', 'install', '--upgrade', path_to_setup])


def force_reinstall_library(lib_name):
    path_to_setup = os.path.join('libs', lib_name)
    if 'fusion' in lib_name:
        path_to_setup = os.path.join(path_to_setup, 'gpu')
    subprocess.run(['pip', 'install', '--force-reinstall', path_to_setup])


def clean_library(lib_name):
    path_to_setup = os.path.join('libs', lib_name)
    if 'fusion' in lib_name:
        path_to_setup = os.path.join(path_to_setup, 'gpu')
    for d in os.listdir(path_to_setup):
        p = os.path.join(path_to_setup, d)
        if os.path.isdir(p) and (d in ['build', 'dist'] or 'egg-info' in d):
            shutil.rmtree(p)
        elif os.path.isfile(p) and d.endswith('.pyx'):
            if os.path.isfile(p.replace('.pyx', '.cpp')):
                os.remove(p)


def main():
    parser = argparse.ArgumentParser(description='Manage libraries')
    parser.add_argument('command',
                        choices=['install', 'uninstall', 'update', 'force-reinstall', 'clean'],
                        type=str,
                        help='Action to perform')
    parser.add_argument('names',
                        nargs='*',
                        type=str,
                        help='Name(s) of librarie(s) to manage')
    parser.add_argument('--cuda_archs',
                        default='5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX',
                        type=str,
                        help='Comma-separated list of CUDA architectures to compile for')
    args = parser.parse_args()

    assert not ('TORCH_CUDA_ARCH_LIST' in os.environ and os.environ['TORCH_CUDA_ARCH_LIST'] != args.cuda_archs), \
        'CUDA architectures mismatch. Got %s, expected %s.' % (os.environ['TORCH_CUDA_ARCH_LIST'], args.cuda_archs)
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.cuda_archs

    libs = get_libraries([name.removeprefix('lib') for name in args.names])
    info = 'All libraries ' if len(libs) > 1 else 'Library '
    if libs:
        if args.command == 'install':
            for lib in libs:
                install_library(lib)
            logger.info(info + 'installed successfully.')
        elif args.command == 'uninstall':
            for lib in libs:
                uninstall_library(lib)
            logger.info(info + 'uninstalled successfully.')
        elif args.command == 'update':
            for lib in libs:
                update_library(lib)
            logger.info(info + 'updated successfully.')
        elif args.command == 'force-reinstall':
            for lib in libs:
                force_reinstall_library(lib)
            logger.info(info + 'reinstalled successfully.')
        elif args.command == 'clean':
            for lib in libs:
                clean_library(lib)
            logger.info(info + 'cleaned successfully.')
    else:
        logger.warning('No librarie(s) found.')


if __name__ == '__main__':
    main()
