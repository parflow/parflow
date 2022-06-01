.. _PorosityExample:

Porosity
~~~~~~~~~~

Here, porosity values are assigned within geounits (specified in
§6.1.4 :ref:`Geometries` above) using one of the methods described
below.

The format for this section of input is:

*list* **Geom.Porosity.GeomNames** [no default] 

This key specifies all of
the geometries on which a porosity will be assigned. These geometries
must cover the entire computational domain.

.. container:: list

   ::

      pfset Geom.Porosity.GeomNames   "background"

.. toctree::
   :maxdepth: 2

   key1
   key2