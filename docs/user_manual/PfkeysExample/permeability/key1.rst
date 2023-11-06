Perm.Conditioning.FileName
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*string* **Perm.Conditioning.FileName** [“NA”] 

This key specifies the name of the file that contains 
the conditioning data. The default string
**NA** indicates that conditioning data is not applicable.

.. container:: list

   ::

      pfset Perm.Conditioning.FileName   "well_cond.txt"

The file that contains the conditioning data is a simple ascii file
containing points and values. The format is:

.. container:: list

   ::

      nlines
      x1 y1 z1 value1
      x2 y2 z2 value2
      .  .  .    .
      .  .  .    .
      .  .  .    .
      xn yn zn valuen

The value of *nlines* is just the number of lines to follow in the file,
which is equal to the number of data points.

The variables *xi,yi,zi* are the real space coordinates (in the units
used for the given parflow run) of a point at which a fixed permeability
value is to be assigned. The variable *valuei* is the actual
permeability value that is known.

Note that the coordinates are not related to the grid in any way.
Conditioning does not require that fixed values be on a grid. The PGS
algorithm will map the given value to the closest grid point and that
will be fixed. This is done for speed reasons. The conditioned turning
bands algorithm does not do this; conditioning is done for every grid
point using the given conditioning data at the location given. Mapping
to grid points for that algorithm does not give any speedup, so there is
no need to do it.

NOTE: The given values should be the actual measured values - adjustment
in the conditioning for the lognormal distribution that is assumed is
taken care of in the algorithms.

The general format for the permeability input is as follows:

*list* **Geom.Perm.Names** [no default] 

This key specifies all of the
geometries to which a permeability field will be assigned. These
geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset GeomInput.Names   "background domain concen_region"