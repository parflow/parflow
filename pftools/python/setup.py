import pathlib
from setuptools import find_packages, setup

# reading the README file
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='pftools',
    version="0.0.3",
    description='A Python package creating an interface with the ParFlow hydrologic model.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/parflow/parflow/tree/master/pftools/python',
    author='HydroFrame',
    author_email='parflow@parflow.org',
    license='BSD',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    keywords=['ParFlow', 'groundwater model', 'surface water model'],
    packages=find_packages(),
    install_requires=['pyyaml==5.3.1'],
    extras_require={
        'all': [
            'imageio>=2.9.0',
            'parflowio'
        ],
        'pfb': [
            'parflowio'
        ],
        'pfsol': [
            'imageio>=2.9.0'
        ]
    }
)
