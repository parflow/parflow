# ParFlow/docs/pf-keys/

This directory contains the files necessary to configure and publish the
documentation of ParFlow keys and Python-PFTools to readthedocs.

### Generating documentation of ParFlow keys

The documentation of ParFlow keys  are generated from the
YAML files located in *ParFlow/pf-keys/definitions*.

To generate the key documentation, run the following command:

        make GenerateKeysRST

This will generate the *keys.rst* file in the ./parflow/ subdirectory. However,
if you ``git push`` changes to the repo, the *conf.py* file will also trigger
a rebuild of the *keys.rst* file. 

### Supporting documentation and tutorials

The following files are written to support the tutorials on readthedocs:

- *parflow/keys_contribution.rst*: Documents the root YAML file structure and how
to add a new key to the ParFlow Python library.
- *python/getting_started.rst*: Directions for installing the Python PFTools module.
- *python/run_script.rst*: Documentation on the anatomy and methods callable within
a ParFlow run script using Python PFTools.
- *tutorials/fs.rst*: Documentation and tutorial on the file system handling functions
within Python PFTools.
- *tutorials/pfb.rst*: Documentation and tutorial on how to work with ParFlow binary (PFB)
 files using Python PFTools.
- *tutorials/solid_files.rst*: Documentation and tutorial on how to generate and read
 solid files with Python PFTools.
- *tutorials/tcl2py.rst*: Documentation and tutorial on how to convert a ParFlow TCL 
script to a Python script to work with Python PFTools.



