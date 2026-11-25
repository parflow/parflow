.. _pftools_installation:

Installation
============

``pftools`` can be installed using pip. This installs our latest stable release with
fully-supported features:

.. code-block::

    pip install pftools

If you would like to set up a virtual environment to install ``pftools``, execute the following commands:

.. code-block::

    python3 -m venv py-env
    source py-env/bin/activate
    pip install pftools

Alternatively, you can create a conda environment and install ``pftools`` as follows (you
can choose a more recent version of Python if desired):

.. code-block::

    conda create -n py-env python=3.13
    conda activate py-env
    pip install pftools

For developers
--------------

Developers and users who need optional tooling can install with the ``[all]`` extras:

.. code-block::

    pip install pftools[all]

.. note::

   On macOS, quote the extras specifier to avoid a shell error:

   .. code-block::

       pip install 'pftools[all]'

The ``[all]`` extras install every optional dependency group below (solid-file
helpers, PDI/HDF5 support, and developer tooling):

- ``[pfsol]``: installs the ``imageio`` package for image processing used in some
  workflows that build ParFlow solid (``.pfsol``) files.
- ``[pdi]``: installs the ``h5py`` package for HDF5 / PDI-related workflows.
- ``[dev]``: installs developer tooling (``twine``, ``black``, and ``gersemi``)
  used for packaging and code formatting.