import pathlib
from setuptools import find_packages, setup

# reading the README file
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='pftools',
    version="0.0.1",
    description='A package to run ParFlow via a Python interface.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/parflow/parflow/tree/master/pftools/python',
    author='HydroFrame',
    author_email='parflow@parflow.org',
    license='BSD',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    keywords=['ParFlow','groundwater model','surface water model'],
    packages=find_packages(),
    install_requires=['pyyaml', 'parflowio']
)
