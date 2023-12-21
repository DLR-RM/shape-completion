from typing import Any
from pathlib import Path
import os
import subprocess

import trimesh


vega_lib_dir = Path(__file__).parent.absolute()
if not os.environ.get('LD_LIBRARY_PATH', False) or str(vega_lib_dir) not in os.environ['LD_LIBRARY_PATH']:
    os.environ['LD_LIBRARY_PATH'] = f'$LD_LIBRARY_PATH:{str(vega_lib_dir)}'


def compute_distance_field(mesh_path: str, resolution: int = 256, **kwargs: Any):
    command = f'{vega_lib_dir}/computeDistanceField {mesh_path} {resolution} {resolution} {resolution}'
    kwarg_list = ['n', 's', 'o', 'm', 'b', 'c', 'e', 'd', 't', 'w', 'W', 'g', 'G', 'r', 'i', 'v', 'p']

    if kwargs.get('w') is not None and kwargs.get('W') is not None:
        kwarg_list.remove('w')

    if kwargs.get('g') is not None and kwargs.get('G') is not None:
        if kwargs.get('G') == 1:
            kwarg_list.remove('G')
        else:
            kwarg_list.remove('g')

    for kwarg in kwarg_list:
        value = kwargs.get(kwarg)
        if value is not None:
            if isinstance(value, bool):
                if value:
                    command += f' -{kwarg}'
            else:
                command += f' -{kwarg} {value}'

    verbose = False
    if kwargs.get('verbose'):
        verbose = True
        print('SDF command:', command)
    subprocess.run(command.split(' '), stdout=None if verbose else subprocess.DEVNULL)
    os.remove(mesh_path)


def compute_marching_cubes(sdf_path: str,
                           output_path: str,
                           **kwargs: Any):
    vega_output_path = os.path.relpath(output_path.replace('.ply', '.obj'))
    command = f'{vega_lib_dir}/isosurfaceMesher {sdf_path} {int(1e5)} {vega_output_path}'

    level = kwargs.get('i')
    if level is not None:
        command += f' -i {level}'

    if kwargs.get('n'):
        command += ' -n'

    verbose = False
    if kwargs.get('verbose'):
        verbose = True
        print('Marching cubes command:', command)

    subprocess.run(command.split(' '), stdout=None if verbose else subprocess.DEVNULL)

    trimesh.load(vega_output_path,
                 force='mesh',
                 process=False,
                 validate=False).export(output_path,
                                        encoding=kwargs.get('encoding', 'binary'),
                                        vertex_normal=kwargs.get('vertex_normal', False),
                                        include_attributes=kwargs.get('include_attributes', False))

    return trimesh.load_mesh(output_path).is_watertight
