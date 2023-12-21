from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

long_description = ''
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    pass

extensions = [
    Extension(
        name="mise",
        sources=["mise.pyx"],
        language="c++",
    )
]

setup(
    name="mise",
    version="1.0",
    description="A C++ and Cython extension for multi-resolution ISO surface extraction.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lars Mescheder',
    author_email='LarsMescheder@gmx.net',
    url='https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh/utils/libmise',
    license='MIT',
    install_requires=[
        'cython',
    ],
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
