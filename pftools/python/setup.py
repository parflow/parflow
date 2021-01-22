from pathlib import Path
from setuptools import find_packages, setup

# reading the README file
HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='pftools',
    version="1.1.0",
    description=('A Python package creating an interface with the ParFlow '
                 'hydrologic model.'),
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
    include_package_data=True,
    extras_require={
        'all': [
            'imageio>=2.9.0',
            'parflowio>=0.0.4'
        ],
        'pfb': [
            'parflowio>=0.0.4'
        ],
        'pfsol': [
            'imageio>=2.9.0'
        ]
    }
)
