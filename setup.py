from setuptools import setup, find_packages

setup(
    name='shape_completion',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'vis_data = visualize.scripts.vis_data:main',
            'vis_inference = visualize.scripts.vis_inference:main',
            'render = visualize.scripts.render:main',
            'process_dataset = dataset.scripts.process_dataset:main',
            'find_pose = process.find_pose:main',
            'make_watertight = process.make_watertight:main',
            'render_kinect = process.render_kinect:main',
            'render_kinect_parallel = process.render_kinect_parallel:main',
            'find_uncertain_regions = process.find_uncertain_regions:main'
        ],
    },
    install_requires=[
        # List your project's dependencies here
    ],
)
