.. _keys_contribution:

******************
Contributing keys
******************

.. _keys_contribution_def:

YAML definitions
=================

The files in this directory are split up into groups to limit their length. Each ParFlow key comprises one or more
tokens, separated by periods. In the YAML files, tokens are set up in a tabbed hierarchical structure, where each
token is nested within the preceding token. Tokens are either static (starting with a capital letter) or dynamic
(denoted by ``.{dynamic_name}``, e.g. ``geom_name`` in ``Geom.geom_name.Lower.X``). Leaf tokens are the tokens where
the value is stored (e.g. ``R`` in ``Process.Topology.R``). All other tokens are referred to as intermediate tokens.

Each token has one or more annotations associated with it, which fall into one of three categories, which are described
below:

.. _keys_contribution_generator_annotations:

1. Generator annotations
-------------------------

The generator uses these annotations to generate the Python library and documentation

.. _keys_contribution_class:

``__class__``
^^^^^^^^^^^^^^

This is for adding dynamically defined tokens. The generator uses the `__class__` name to reference the
location of a dynamically defined token. The `__class__` names usually end in `Item` to denote a dynamic token,
e.g. `CycleItem` for the `.{cycle_name}` token in the key `Cycle.cycle_name.Names`.

.. _keys_contribution_from:

``__from__``
^^^^^^^^^^^^^

This includes the source path of the dynamically defined token referenced in ``__class__``. For example,

.. code-block:: python3

    BCPressure:

        .{interval_name}:
            __class__: BCPressureIntervalItem
            __from__: /Cycle/{CycleItem}/Names

Here, ``BCPressureIntervalItem`` is the dynamically defined ``.{interval_name}`` token. The values of the
``Cycle.cycle_name.Names`` key generate these ``.{interval_name}`` tokens. The path to the ``Cycle.cycle_name.Names``
key is ``/Cycle/{CycleItem}/Names``.

.. _keys_contribution_rst:

``__rst__``
^^^^^^^^^^^^

This contains details to support the documentation. Arguments for this include:

- ``name:`` {string}: This argument will change the name of the key as it appears in the documentation.
- ``skip:`` {no arguments}: This will cause the key to not print in the documentation. This does not affect nested tokens.
- ``warning:`` {string}: This argument will add a special warning message to the documentation. This should be used for special cases, such as when a key must be set differently in Python as opposed to a TCL script.

.. _keys_contribution_prefix:

``__prefix__``
^^^^^^^^^^^^^^^

This handles the tokens for key names with integers as tokens (e.g. ``Cell.0.dzScale.Value``). Since Python does not
recognize integers as a valid variable name, the user must specify a prefix to the integer. This can be any alphabetical
character (upper or lower case) or an underscore. The specified prefix must be used to set the token within the key. For
example, the prefix for ``Cell.0.dzScale.Value`` is an underscore, so you must define the key as ``Cell._0.dzScale.Value``.

.. _keys_contribution_key_annotations:

2. Key annotations
-------------------

These annotations apply to the key itself, assisting documentation

.. _keys_contribution_help_doc:

``help``, ``__doc__``
^^^^^^^^^^^^^^^^^^^^^

This contains the documentation for the key. ``help`` is used for leaf tokens, and ``__doc__`` is used for intermediate
tokens.

.. _keys_contribution_value:

``__value__``
^^^^^^^^^^^^^^

This annotation applies to intermediate tokens that contain a value, but are not a leaf token (e.g. ``Solver``). This will
be treated as if it were a leaf token, including the value annotations that apply to the intermediate token.

.. _keys_contribution_value_annotations:

3. Value annotations
---------------------

These annotations apply to the value set to the key.

.. _keys_contribution_domains:

``domains``
^^^^^^^^^^^^

This defines the domains that constrain the value of the key. The domains must include one or more of the following:

- ``AddedInVersion``: This is for keys that have been added in a known version of ParFlow. This takes a string argument of the ParFlow version when the key was added, e.g. ``'3.6.0'``.

- ``AnyString``: This is for keys that must be a string. There are no arguments for this domain.

- ``BoolDomain``: This is for keys that must be ``True`` or ``False``. There are no arguments for this domain.

- ``DeprecatedInVersion``: This is for keys that have been or will be deprecated in a known version of ParFlow. This takes a string argument of the ParFlow version when the key has been or will be deprecated, e.g., ``'3.6.0'``.

- ``DoubleValue``: This is for keys that must be a double. It takes up to two arguments: ``min_value`` for the minimum value, and ``max_value`` for the maximum value. Keys with a DoubleValue domain can also be integers.

- ``EnumDomain``: This is for values that must be one of an enumerated list. This takes one argument, ``enum_list``, that includes the list of acceptable values for the key. To accommodate instances where new options are added in new versions of ParFlow, ``enum_list`` can take an argument of a ParFlow version, which would include the list of acceptable values beginning with the specified version. See the ``EnumDomain`` for the ``Patch.{patch_name}.BCPressure.Type`` (*bconditions.yaml*) for an example.

- ``IntValue``: This is for keys that must be an integer. It takes up to two arguments: ``min_value`` for the minimum value, and ``max_value`` for the maximum value.

- ``MandatoryValue``: This is for keys that must be set for a ParFlow run. ``MandatoryValue`` does not take any arguments.

- ``RemovedInVersion``: This is for keys that have been or will be removed in a known version of ParFlow. This takes a string argument of the ParFlow version when the key has been or will be removed, e.g. ``'3.6.0'``.

- ``RequiresModule``: This is for keys that must have a particular module installed or compiled to be a valid key (e.g., ``Solver.CLM....``). This takes an argument of the required module in all caps, e.g., ``RequiresModule: NETCDF``.

- ``ValidFile``: This is for keys which reference file names to make sure that the file exists. It can take two arguments: ``working_directory``, for which you can specify the absolute path of the directory where your file is stored, ``path_prefix_source``, for which you can specify the path to a key that defines the path
    to the file (e.g. ``Solver.CLM.MetFile``). If no arguments are provided, it will check your current working directory for the file.

.. _keys_contribution_handlers:

``handlers``
^^^^^^^^^^^^^

This will transform inputs or help generate dynamically defined tokens within other keys based on the provided value for the key. Each
argument is an updater that specifies where and how the value is used to create other tokens. An example from phase.yaml
is below:

.. code-block:: python3

            Phase:
                Names:
                    handlers:
                        PhaseUpdater:
                            type: ChildrenHandler
                            class_name: PhaseNameItem
                            location: .

``PhaseUpdater`` is the name of the handler. The arguments for the handler include ``type``, ``class_name``, and ``location``.
The most common option for ``type`` is ``ChildrenHandler``. ``class_name`` corresponds to the ``__class__`` annotation of the
dynamic token. In this example, ``PhaseNameItem`` is the ``__class__`` of the dynamic token ``.{phase_name}``. ``location`` is
the location of the token referenced in ``class_name``. In this example, the Names token in ``Phase.Names`` is on the same
level as the ``.{phase_name}`` in ``Phase.phase_name``. This can also be an absolute path. See ``handlers.py`` for more on the other handlers.

.. _keys_contribution_ignore:

``ignore``
^^^^^^^^^^^

Skip field exportation but allow to set other keys from it in a more convenient manner using some handler.

.. code-block:: yaml

    Solver:
        CLM:
            Input:
                Timing:
                    StartDate:
                      help: >
                        [Type: string] Helper property that will set StartYear/StartMonth/StartDay
                      ignore: _not_exported_
                      handlers:
                        FieldsUpdater:
                          type: SplitHandler
                          separator: /
                          convert: int
                          fields:
                            - StartYear
                            - StartMonth
                            - StartDay

.. _keys_contribution_steps:

Steps to add a new key
=======================

1. Select the yaml file that most closely matches the key that you want to add. If your key is a token nested within an
existing key, be sure to find which yaml file includes the parent token(s). For example, if you wanted to add the key
``Solver.Linear.NewKey``, you would add it within the file *solver.yaml*.

2. Open the yaml file and navigate to the level within the hierarchy where you want to put your key. The structure of
the yaml files is designed to be easy to follow, so it should be easy to find the level where you'd like to add your
key. The indentation of these files is two spaces. Using our ``Solver.Linear.NewKey`` example, ``Solver`` is at the far
left, ``Linear`` is two spaces (one tab) in, and you would add ``NewKey`` two more spaces in (two tabs). We suggest copying
and pasting an existing key from the same level to make sure it's correct.

3. Fill in the details of your key. Again, this format is designed to be readable, so please refer to examples in the
yaml files to guide you. The details you can include are listed in the section above.

4. Regenerate the Python keys using ``make GeneratePythonKeys``.

You should see a longer message indicating an update that lists the overlapping classes, including the line ``Defined ##
fields were found``.

5. Test your new key. If you have an input script with the new key, you can run that to check whether it's working.
