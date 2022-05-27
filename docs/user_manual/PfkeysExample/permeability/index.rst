.. _PermeabilityExample:

Permeability
~~~~~~~~~~~~~~

In this section, permeability property values are assigned to grid
points within geometries (specified in §6.1.4 :ref:`Geometries` above)
using one of the methods described below. Permeabilities are assumed to
be a diagonal tensor with entries given as,

.. math::

   \left( 
   \begin{array}{ccc}
   k_x({\bf x}) & 0 & 0 \\
   0 & k_y({\bf x}) & 0 \\
   0 & 0 & k_z({\bf x}) 
   \end{array} \right) 
   K({\bf x}),

where :math:`K({\bf x})` is the permeability field given below.
Specification of the tensor entries (:math:`k_x, k_y` and :math:`k_z`)
will be given at the end of this section.

The random field routines (*turning bands* and *pgs*) can use
conditioning data if the user so desires. It is not necessary to use
conditioning as ParFlow automatically defaults to not use conditioning
data, but if conditioning is desired, the following key should be set.

.. toctree::
   :maxdepth: 2

   key1
   key2
   key3