********************************************************************************
Tables for Subsurface Parameters
********************************************************************************


================================================================================
Introduction
================================================================================

ParFlow domains with complex geology often involve many lines in the input script, which lengthens the script and makes it more cumbersome to navigate. Python PFTools makes it easy to do the following:

- Load in a table of your subsurface properties
- Export a table of the subsurface properties
- Load a database of common soil and geologic properties to set up your domain

================================================================================
Usage of ``SubsurfacePropertiesBuilder``
================================================================================

First, we'll show some usage examples of loading tables of parameters within a ParFlow Python script:

.. code-block:: python3

    from parflow import Run
    from parflow.tools.builders import SubsurfacePropertiesBuilder

    table_test = Run("table_test", __file__)

    table_test.GeomInput.Names = 'box_input indi_input'
    table_test.GeomInput.indi_input.InputType = 'IndicatorField'
    table_test.GeomInput.indi_input.GeomNames = 's1 s2 s3 s4 g1 g2 g3 g4'

    # First example: table as in-line text
    soil_properties = '''
    # ----------------------------------------------------------------
    # Sample header
    # ----------------------------------------------------------------
    key     Perm   Porosity   RelPermAlpha  RelPermN  SatAlpha  SatN    SatSRes    SatSSat

    s1      0.26   0.375      3.548         4.162     3.548     4.162   0.000001   1
    s2      0.04   -          3.467         2.738     -         2.738   0.000001   1
    s3      0.01   0.387      2.692         2.445     2.692     2.445   0.000001   1
    s4      0.01   0.439      -             2.659     0.501     2.659   0.000001   1
    '''

    # Creating object and assigning the subsurface properties
    SubsurfacePropertiesBuilder(table_test) \
      .load_txt_content(soil_properties) \              # loading the in-line text
      .load_csv_file('geologic_properties_123.csv') \   # loading external csv file
      .assign('g3', 'g4') \                             # assigns properties of unit 'g3' to unit 'g4'
      .apply() \                                        # setting keys based on loaded properties
      .print_as_table()                                 # printing table of loaded properties

At the top of the script, we import the ``SubsurfacePropertiesBuilder`` class from ``parflow.tools.builders``. Then,
after specifying the ``GeomInput.indi_input.GeomNames``, we include a table for the soil properties as an in-line text variable.
Note that the ``GeomInput.indi_input.GeomNames`` are in the first column (``key``), and the parameter names are across the top.
These parameter names don't have to match the key names exactly, but they have to be similar. We will explain this later.

We load the ``soil_properties`` text by calling the ``load_txt_content`` method on the ``SubsurfacePropertiesBuilder`` object.
To load the geologic properties for the geometric units *g1*, *g2*, and *g3*, we call ``load_csv_file`` to load an external csv file.
That now leaves one unit, *g4*, that needs properties. We use the ``assign`` method to assign properties to unit *g4* from the
properties of unit *g3*. Now that all the geometric units have properties, we call ``apply`` to set the appropriate keys.
The ``print_as_table`` method prints out the subsurface properties for each unit. Executing this example will result in a table that looks something like this:

.. code-block:: bash

    key  Perm  Porosity  RelPermAlpha  RelPermN  SatAlpha  SatN   SRes   SSat
    s1   0.26  0.375     3.548         4.162     3.548     4.162  1e-06  1.0
    s2   0.04  -         3.467         2.738     -         2.738  1e-06  1.0
    s3   0.01  0.387     2.692         2.445     2.692     2.445  1e-06  1.0
    s4   0.01  0.439     -             2.659     0.501     2.659  1e-06  1.0
    g1   0.26  0.375     3.548         4.162     3.548     4.162  1e-06  1.0
    g2   0.04  -         3.467         2.738     -         2.738  1e-06  1.0
    g3   0.01  0.387     2.692         2.445     2.692     2.445  1e-06  1.0
    g4   0.01  0.387     2.692         2.445     2.692     2.445  1e-06  1.0

================================================================================
Table formatting for importing
================================================================================

Let's have another look at the in-line table from the usage example above:

.. code-block:: python3

    soil_properties = '''
    # ----------------------------------------------------------------
    # Sample header
    # ----------------------------------------------------------------
    key     Perm   Porosity   RelPermAlpha  RelPermN  SatAlpha  SatN    SatSRes    SatSSat

    s1      0.26   0.375      3.548         4.162     3.548     4.162   0.000001   1
    s2      0.04   -          3.467         2.738     -         2.738   0.000001   1
    s3      0.01   0.387      2.692         2.445     2.692     2.445   0.000001   1
    s4      0.01   0.439      -             2.659     0.501     2.659   0.000001   1
    '''

These tables can be formatted in a number of different ways. Here are several considerations:

- Any blank rows or rows beginning with ``#`` are ignored in processing.
- Delimiters can be either commas or spaces.
- Table orientation does not matter (i.e., whether the field names are across the first row or down the first column). The only requirement is for that the top-left entry be ``key`` or one of its aliases.
- The table does not have to be completely filled. As shown here, blank property values must be designated by a hyphen.
- To properly process the table and map to the correct keys, the field names (including ``key``) must be one of several possible aliases. The aliases are listed in `this yaml file <https://github.com/grapp1/parflow/blob/py-input/pftools/python/parflow/tools/ref/table_keys.yaml>`_ that is included in the Python PFTools. These aliases include the exact end of the key name (e.g., ``Perm.Value`` as opposed to the alias ``Perm``), so when in doubt, you can use the exact name.

================================================================================
Default database loading
================================================================================

We have added several databases of commonly used parameters for different soil and geologic units to provide some helpful guidance. To load these database, you can simply call the ``load_default_properties`` method on the ``SubsurfacePropertiesBuilder`` object.
The available databases in the Python PFTools package can be found `in the "subsurface_*.txt" files here. <https://github.com/parflow/parflow/tree/master/pftools/python/parflow/tools/ref>`_
You can load any of the databases into your ``SubsurfacePropertiesBuilder`` object by passing in the ``database`` argument, which is the latter part of the database file name (e.g. "subsurface_conus_1.txt" can be loaded by calling ``load_default_properties('conus_1')``).
The default database is from `Maxwell and Condon (2016). <https://science.sciencemag.org/content/353/6297/377>`_ Note that the parameters in the databases are all in the default ParFlow units of meters and hours.

----

Below is an example of how to use the default database importer on the ``Run`` object ``db_test``:

.. code-block:: python3

    # setting GeomNames
    db_test.GeomInput.Names = 'box_input indi_input'
    db_test.GeomInput.box_input.InputType = 'Box'
    db_test.GeomInput.box_input.GeomName = 'domain'
    db_test.GeomInput.indi_input.InputType = 'IndicatorField'
    db_test.GeomInput.indi_input.GeomNames = 's1 s2 g2'

    # setting dictionary for mapping from database properties (i.e. the keys of map_dict)
    # to the different subsurface units (i.e. the values of map_dict)
    map_dict = {
      'bedrock_1': ['domain', 'g2'],
      'sand': 's1',
      'loamy_sand': 's2'
    }

    SubsurfacePropertiesBuilder(db_test)\
      .load_default_properties() \
      .assign(mapping=map_dict) \
      .apply() \
      .print_as_table()

The dictionary ``map_dict`` maps the database properties to the subsurface units in your ``Run`` object.
Note that the properties from the database unit ``bedrock_1`` are applied to both ``domain`` and ``g2``. If you are assigning a database unit to multiple ``GeomNames``, these must be input as a list, as shown.
This will print the following:

.. code-block:: bash

    key     Perm     Porosity  SRes   RelPermAlpha  RelPermN
    domain  0.005    0.33      0.001  1.0           3.0
    g2      0.005    0.33      0.001  1.0           3.0
    s1      0.269    0.38      0.14   3.55          4.16
    s2      0.0436   0.39      1.26   3.47          2.74

================================================================================
Full API for ``SubsurfacePropertiesBuilder``
================================================================================

1. ``SubsurfacePropertiesBuilder(run=None)``: Instantiates a ``SubsurfacePropertiesBuilder`` object. If the optional ``Run`` object ``run`` is given, it will use the subsurface units in ``run`` for later application. ``run`` must be provided as an argument either here or when calling the ``apply()`` method (see below).
2. ``load_csv_file(tableFile, encoding='utf-8-sig')``: Loads a comma-separated (csv) file to your ``SubsurfacePropertiesBuilder`` object. The default text encoding format is ``utf-8-sig``, which should translate files generated from Microsoft Excel.
3. ``load_txt_file(tableFile, encoding='utf-8-sig')``: Loads a text file to your ``SubsurfacePropertiesBuilder`` object. The default text encoding format is ``utf-8-sig``.
4. ``load_txt_content(txt_content)``: Loads in-line text to your ``SubsurfacePropertiesBuilder`` object.
5. ``load_default_properties(database='conus_1')``: Loads one of several databases of subsurface properties. The default, ``conus_1``, is from `Maxwell and Condon (2016). <https://science.sciencemag.org/content/353/6297/377>`_
6. ``assign(old=None, new=None, mapping=None)``: Assigns properties to the ``new`` subsurface unit using the properties from the ``old`` subsurface unit. Alternatively, a dictionary (``mapping``) can be passed in as an argument, which should have the keys as the ``old`` units, and the values as the ``new`` units. If an ``old`` unit will apply to multiple ``new`` units, the ``new`` units need to be passed in as a list.
7. ``apply(run=None, name_registration=True)``: Applies the loaded subsurface properties to the subsurface units in the ``Run`` object ``run``. If ``run`` is not provided here, the user must provide the ``run`` argument when instantiating the ``SubsurfacePropertiesBuilder``object. If ``name_registration`` is set to ``True``, it will add the subsurface unit names (e.g., *s1*, *s2* from the example above) to the list of unit names for each property (e.g., setting  ``Geom.Perm.Names = 's1 s2 s3 s4'``), and set the ``addon`` keys not associated with a specific unit (e.g., ``Phase.RelPerm.Type``).
8. ``print()``: Prints out the subsurface parameters for all subsurface units in a hierarchical format.
9. ``print_as_table(props_in_header=True, column_separator='  ')``: Prints out the subsurface parameters for all subsurface units in a table format. ``props_in_header`` will print the table with the property names as column headings if set to ``True``, or as row headings if set to ``False``.

================================================================================
Exporting subsurface properties
================================================================================

It is often useful to have a table of the subsurface properties assigned to various subsurface units during a run. As mentioned in the `run script API <https://grapp1parflow.readthedocs.io/en/latest/python/run_script.html#full-api>`_,
you can write out a table of the subsurface properties by calling the ``write_subsurface_table`` method on your ``Run`` object.

----

For example, try adding the following line just above the ``run()`` method call in the ``default_richards.py`` Python test, with the name of the output file passed in as the ``file_name`` argument:

.. code-block:: python3

    drich.write_subsurface_table(file_name='def_richards_subsurf.txt')

If you do not provide ``file_name``, the default file will be a *.csv* file with the name of your run and *subsurface*. In this case, the default file would be *default_richards_subsurface.csv*.
Execute the Python script, and you should see the output file *def_richards_subsurf.txt* containing the following:

.. code-block:: bash

  key         Perm  Porosity  SpecStorage  RelPermAlpha  RelPermN  SatAlpha  SatN  SRes  SSat
  domain      -     -         0.0001       0.005         2.0       0.005     2.0   0.2   0.99
  background  4.0   1.0       -            -             -         -         -     -     -

See that it only prints out the properties that are explicitly assigned to each of the subsurface units ``domain`` and ``background``.

================================================================================
Examples
================================================================================

Full examples of the ``SubsurfacePropertiesBuilder`` can be found in the *new_features* subdirectory of the ParFlow Python tests.

- *default_db*: Loading the default database and mapping the database units to subsurface units in the current run.
- *tables_LW*: Showing multiple ways to load tables to replace the subsurface property definition keys in the Little Washita test script.
