Geom.geometry_name.Perm.Type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*string* **Geom.geometry_name.Perm.Type** [no default] 

This key specifies which method is to be used to 
assign permeability data to the named
geometry, *geometry_name*. It must be either **Constant**,
**TurnBands**, **ParGuass**, or **PFBFile**. The **Constant** value
indicates that a constant is to be assigned to all grid cells within a
geometry. The **TurnBand** value indicates that Tompson’s Turning Bands
method is to be used to assign permeability data to all grid cells
within a geometry [TAG89]. The **ParGauss** value
indicates that a Parallel Gaussian Simulator method is to be used to
assign permeability data to all grid cells within a geometry. The
**PFBFile** value indicates that premeabilities are to be read from the
“ParFlow Binary” file. Both the Turning Bands and Parallel Gaussian
Simulators generate a random field with correlation lengths in the
:math:`3` spatial directions given by :math:`\lambda_x`,
:math:`\lambda_y`, and :math:`\lambda_z` with the geometric mean of the
log normal field given by :math:`\mu` and the standard deviation of the
normal field given by :math:`\sigma`. In generating the field both of
these methods can be made to stratify the data, that is follow the top
or bottom surface. The generated field can also be made so that the data
is normal or log normal, with or without bounds truncation. Turning
Bands uses a line process, the number of lines used and the resolution
of the process can be changed as well as the maximum normalized
frequency :math:`K_{\rm max}` and the normalized frequency increment
:math:`\delta K`. The Parallel Gaussian Simulator uses a search
neighborhood, the number of simulated points and the number of
conditioning points can be changed.

.. container:: list

   ::

      pfset Geom.background.Perm.Type   Constant