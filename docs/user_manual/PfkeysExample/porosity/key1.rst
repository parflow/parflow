Geom.Porosity.Type
^^^^^^^^^^^^^^^^^^^

*string* **Geom.geometry_name.Porosity.Type** [no default] 

This key specifies which method is to be used to assign porosity data to the
named geometry, *geometry_name*. The only choice currently available is
**Constant** which indicates that a constant is to be assigned to all
grid cells within a geometry.

.. container:: list

   ::

      pfset Geom.background.Porosity.Type   Constant