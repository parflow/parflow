********************************************************************************
Tables for Subsurface Parameters
********************************************************************************


================================================================================
Introduction
================================================================================

ParFlow domains with complex geology often involve many lines in the input script, which lengthens the script and makes it more cumbersome to navigate.
Wouldn't it be easier to load in a table of your subsurface properties?
Wouldn't it also be nice to be able to load a database of common soil and geologic properties to set up your domain? Python-PFTools has a way to do it all.

================================================================================
Usage
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
      .assign('g4', 'g3') \                             # assigns properties of unit 'g3' to unit 'g4'
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
The ``print_as_table`` method prints out the subsurface properties for each unit.

================================================================================
Table formatting
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
- Table orientation does not matter (i.e., whether the field names are across the first row or
down the first column). The only requirement is for that the top-left entry be ``key`` or one of its aliases.
- The table does not have to be completely filled. As shown here, blank property values must be designated by a hyphen.
- To properly process the table and map to the correct keys, the field names (including ``key``) must be one of
several possible aliases. The aliases are listed in a yaml file that is included in the Python PFTools, which
can also be found `here. <https://github.com/grapp1/parflow/blob/py-input/pftools/python/parflow/tools/ref/table_keys.yaml>`_
These aliases include the exact end of the key name (e.g., ``Perm.Value`` as opposed to the alias ``Perm``), so when in
doubt, you can use the exact name.

================================================================================
Default database loading
================================================================================

We have added a database of commonly used parameters for different soil and geologic units to provide some helpful guidance.
This table is from `Maxwell and Condon (2016). <https://science.sciencemag.org/content/353/6297/377>`_
The table in the Python PFTools package can be found `here. <https://science.sciencemag.org/content/353/6297/377>`_
To load this database, you can simply call the ``load_default_properties`` method on the ``SubsurfacePropertiesBuilder`` object.

================================================================================
Full API
================================================================================

1. ``load_csv_file(tableFile, encoding='utf-8-sig')``: Loads a comma-separated (csv) file to your ``SubsurfacePropertiesBuilder`` object.
The default text encoding format is ``utf-8-sig``, which should translate files generated from Microsoft Excel.
2. ``load_txt_file(tableFile, encoding='utf-8-sig')``: Loads a text file to your ``SubsurfacePropertiesBuilder`` object.
The default text encoding format is ``utf-8-sig``.
3. ``load_txt_content(txt_content)``: Loads in-line text to your ``SubsurfacePropertiesBuilder`` object.
4. ``load_default_properties()``: Loads the table of the default subsurface properties from Maxwell et al. (2016).
5. ``assign(new=None, old=None, mapping=None)``: Assigns properties to the ``new`` subsurface unit using the
properties from the ``old`` subsurface unit. Alternatively, a dictionary (``mapping``) can be passed in as an argument, which
should have the keys as the ``new`` units, and the values as the ``old`` units.
6. ``apply(name_registration=True)``: Applies the loaded subsurface properties to the subsurface units. If
``name_registration`` is set to ``True``, it will add the subsurface unit names (e.g., *s1*, *s2* from the
example above) to the list of unit names for each property (e.g., setting  ``Geom.Perm.Names = 's1 s2 s3 s4'``), and set
the ``addon`` keys not associated with a specific unit (e.g., ``Phase.RelPerm.Type``).
7. ``print()``: Prints out the subsurface parameters for all subsurface units in a hierarchical format.
8. ``print_as_table(props_in_header=True, column_separator='  ')``: Prints out the subsurface parameters for all
subsurface units in a table format. ``props_in_header`` will print the table with the property names as column headings
if set to ``True``, or as row headings if set to ``False``.

================================================================================
Examples
================================================================================

Full examples of the ``SubsurfacePropertiesBuilder`` can be found in the *new_features* subdirectory of the ParFlow Python tests.
- *default_db*: Loading the default database and mapping the database units to subsurface units in the current run.
- *tables_LW*: Showing multiple ways to load tables to replace the subsurface property definition keys in the Little Washita test script.
