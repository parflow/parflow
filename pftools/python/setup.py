import pathlib
from setuptools import find_packages, setup

# reading the README file
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='parflow',
    version="0.1.0",
    description='A package to run ParFlow via a Python interface.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/grapp1/kw-intern/tree/master/parflow-integration/pftools/python',
    author='Kitware, Inc.',
    license='BSD',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    keywords=['ParFlow','groundwater model','surface water model'],
    packages=['parflow','parflow.tools','parflow.tools.database'],
    install_requires=['pyyaml']
)
