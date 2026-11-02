.. _domain_builder:

Domain definition helpers
==========================

.. _domain_builder_intro:

Introduction
-------------

One of ParFlow's strengths is its customizability; you can practically define any type of hydrologic problem with it.
One of the downsides of that, however, is that setting all the keys can be cumbersome, especially when starting a run from scratch.
With the new ``DomainBuilder``, Python-PFTools helps condense the setting of keys for many common problem definitions.

.. _domain_builder_usage:

Usage of ``DomainBuilder``
---------------------------

First, we'll show some usage examples of loading tables of parameters within a ParFlow Python script:

.. code-block:: python3

    from parflow import Run
    from parflow.tools.builders import DomainBuilder

    LW_Test = Run("LW_Test", __file__)

    # ----------------------------------------------------------------------------

    bounds = [
        0.0, 41000.0,
        0.0, 41000.0,
        0.0, 100.0
    ]

    domain_patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'
    zero_flux_patches = 'x_lower x_upper y_lower y_upper z_lower'

    DomainBuilder(LW_Test) \
        .no_wells() \
        .no_contaminants() \
        .water('domain') \
        .variably_saturated() \
        .box_domain('box_input', 'domain', bounds, domain_patches) \
        .homogeneous_subsurface('domain', specific_storage=1.0e-5, isotropic=True) \
        .zero_flux(zero_flux_patches, 'constant', 'alltime') \
        .slopes_mannings('domain', slope_x='LW.slopex.pfb', slope_y='LW.slopey.pfb', mannings=5.52e-6) \
        .ic_pressure('domain', patch='z_upper', pressure='press.init.pfb')

In this example, the 10 lines associated with the instantiation of the ``DomainBuilder`` class generate about 70 keys!
As is possible with any other key setting, you can always overwrite the keys as necessary; the ``DomainBuilder`` is designed to help you get started.
Once you instantiate the ``DomainBuilder`` object on a ``Run`` object, each method will set various keys with the given arguments.
