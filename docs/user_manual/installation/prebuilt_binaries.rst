Installing ParFlow from Pre-built Binaries (Linux and MacOS)
============================================================

Instructions for setting up ParFlow locally with pre-built MacOS and Linux binaries
-----------------------------------------------------------------------------------

Please note that this feature is under development. Future ParFlow releases with include pre-built binaries as part of the release. More information will be provided in the `ParFlow Google Group <https://groups.google.com/g/parflow?pli=1>`_.

MacOS
~~~~~

1. Download the pre-built binaries from the `MacOS artifact <https://github.com/parflow/parflow/actions/runs/26672439563/artifacts/7304739867>`_.
2. Navigate to your ``Downloads`` folder, and double-click on the zip archive ``parflow-macos-arm64.zip``. The resulting ``install`` directory is where ParFlow will be located on your system.
3. Open your terminal, and run the following command from the ``Downloads`` directory:

.. code-block:: bash

   xattr -dr com.apple.quarantine install/

4. Set the ``PARFLOW_DIR`` environment variable to your install location. For example, in a Python script, add the line

.. code-block:: python

   os.environ['PARFLOW_DIR'] = '/Users/<your_username>/Downloads/install'

or in a shell script, add the line

.. code-block:: bash

   export PARFLOW_DIR=/Users/<your_username>/Downloads/install

5. Note that in addition to these pre-built binaries, you will also need a Python environment containing ``pftools``. This package can be installed with

.. code-block:: bash

   pip install pftools

Linux
~~~~~

1. Download the pre-built binaries from the `Linux artifact <https://github.com/parflow/parflow/actions/runs/26650477704/artifacts/7297049803>`_. You will get a file like ``parflow-pr-742-7f54d79e4655c3dd2abdfd5bb12759cfa43a3adb-linux-x86_64.tar.gz.zip``
2. Run the following commands:

.. code-block:: bash

   unzip parflow-pr-742-7f54d79e4655c3dd2abdfd5bb12759cfa43a3adb-linux-x86_64.tar.gz.zip
   tar -xvf parflow-pr-742-7f54d79e4655c3dd2abdfd5bb12759cfa43a3adb-linux-x86_64.tar.gz

You will end up with a directory called ``release-install``. This is where your local ParFlow installation will be located.

3. Set the ``PARFLOW_DIR`` environment variable to your install location. For example, in a Python script, add the line

.. code-block:: python

   os.environ['PARFLOW_DIR'] = '/path/to/release-install'

or in a shell script, add the line

.. code-block:: bash

   export PARFLOW_DIR=/path/to/release-install

Alternatively, you can source the ``parflow-env.sh`` script. This will set the PARFLOW_DIR variable in your shell.

.. code-block:: bash

   source parflow-env.sh 

4. Note that in addition to these pre-built binaries, you will also need a Python environment containing ``pftools``. This package can be installed with

.. code-block:: bash

   pip install pftools
