.. _ParFlow Files:

ParFlow Files
=============

In this chapter, we discuss the various file formats used in ParFlow. To
help simplify the description of these formats, we use a pseudocode
notation composed of *fields* and *control constructs*.

A field is a piece of data having one of the *field types* listed in
Table `5.1 <#table-field-types>`__ (note that field types may have one
meaning in ASCII files and another meaning in binary files).

.. container::
   :name: table-field-types

   .. table:: Field types.

      +-------------------------+----------------+------------------------+
      | field type              | ASCII          | binary                 |
      +=========================+================+========================+
      | ``integer``             | ``integer``    | XDR ``integer``        |
      +-------------------------+----------------+------------------------+
      | ``real``                | ``real``       |                        |
      +-------------------------+----------------+------------------------+
      | ``string``              | ``string``     |                        |
      +-------------------------+----------------+------------------------+
      | ``double``              |                | IEEE 8 byte ``double`` |
      +-------------------------+----------------+------------------------+
      | ``float``               |                | IEEE 4 byte ``float``  |
      +-------------------------+----------------+------------------------+


Fields are denoted by enclosing the field name with a ``<`` on the left 
and a ``>`` on the right. The field name is composed of alphanumeric 
characters and underscores (``_``). In the defining entry of a field, 
the field name is also prepended by its field type and a ``:``. 
The control constructs used in our pseudocode have the keyword 
names ``FOR``, ``IF``, and ``LINE``, and the beginning and end of 
each of these constructs is delimited by the keywords ``BEGIN`` 
and ``END``.

The ``FOR`` construct is used to describe repeated input format 
patterns. For example, consider the following file format:

.. container:: list

   ::

      <integer : num_coordinates>
      FOR coordinate = 0 TO <num_coordinates> - 1
      BEGIN
         <real : x>  <real : y>  <real : z>
      END

The field ``<num_coordinates>`` is an integer specifying the number of coordinates to 
follow. The ``FOR`` construct indicates that ``<num_coordinates>`` entries follow, 
and each entry is composed of the three real fields, ``<x>``, ``<y>``, 
and ``<z>``. Here is an example of a file with this format:

.. container:: list

   ::

      3
      2.0 1.0 -3.5
      1.0 1.1 -3.1
      2.5 3.0 -3.7

The ``IF`` construct is actually an ``IF/ELSE`` construct, and is used to describe input 
format patterns that appear only under certain circumstances. For example, consider 
the following file format:

.. container:: list

   ::

      <integer : type>
      IF (<type> = 0)
      BEGIN
         <real : x>  <real : y>  <real : z>
      END
      ELSE IF (<type> = 1)
      BEGIN
         <integer : i>  <integer : j>  <integer : k>
      END

The field ``<type>`` is an integer specifying the “type” of input to 
follow. The ``IF`` construct indicates that if ``<type>`` has value 0, 
then the three real fields, ``<x>``, ``<y>``, and ``<z>``, follow. 
If ``<type>`` has value 1, then the three integer 
fields, ``<i>``, ``<j>``, and ``<k>``, follow. Here is an example 
of a file with this format:

.. container:: list

   ::

      0
      2.0 1.0 -3.5

The ``LINE`` construct indicates fields that are on the same line of 
a file. Since input files in ParFlow are all in “free format”, it is 
used only to describe some output file formats. For example, consider 
the following file format:

.. container:: list

   ::

      LINE
      BEGIN
         <real : x>
         <real : y>
         <real : z>
      END

The ``LINE`` construct indicates that the three real 
fields, ``<x>``, ``<y>``, and ``<z>``
, are all on the same line. Here is an example of a file 
with this format:

.. container:: list

   ::

      2.0 1.0 -3.5

Comment lines may also appear in our file format pseudocode. All text
following a ``#`` character is a comment, and is not part of the file format.

.. _Main Input Files (.tcl, .py, .ipynb):

Main Input Files (.tcl, .py, .ipynb)
------------------------------------

The main ParFlow input file can be a Python script, a TCL script, or a Jupyter Notebook.  For more advanced users,
the notebook or scripting environment provides a lot of flexibility and means you can very easily create programs to run
ParFlow. A simple example is creating a loop to run several hundred
different simulations using different seeds to the random field
generators. This can be automated from within the ParFlow input file. The input structure for these files is given in the :ref:`ParFlow Input Keys` chapter.

.. _ParFlow Binary Files (.pfb):

ParFlow Binary Files (.pfb)
---------------------------

The ``.pfb`` file format is a binary file format which is used to store ParFlow 
grid data. It is written as BIG ENDIAN binary bit ordering :cite:p:`endian`. The format 
for the file is:

.. container:: list

   ::

      <double : X>    <double : Y>    <double : Z>
      <integer : NX>  <integer : NY>  <integer : NZ>
      <double : DX>   <double : DY>   <double : DZ>

      <integer : num_subgrids>
      FOR subgrid = 0 TO <num_subgrids> - 1
      BEGIN
         <integer : ix>  <integer : iy>  <integer : iz>
         <integer : nx>  <integer : ny>  <integer : nz>
         <integer : rx>  <integer : ry>  <integer : rz>
         FOR k = iz TO iz + <nz> - 1
         BEGIN
            FOR j = iy TO iy + <ny> - 1
            BEGIN
               FOR i = ix TO ix + <nx> - 1
               BEGIN
                  <double : data_ijk>
               END
            END
         END
      END

.. _ParFlow Binary Files (.c.pfb):

ParFlow CLM Single Output Binary Files (.c.pfb)
-----------------------------------------------

The ``.pfb`` file format is a binary file format which is used to 
store ``CLM`` output data in a single file. It is written as 
BIG ENDIAN binary bit ordering :cite:p:`endian`. The format for the file is:

.. container:: list

   ::

      <double : X>    <double : Y>    <double : Z>
      <integer : NX>  <integer : NY>  <integer : NZ>
      <double : DX>   <double : DY>   <double : DZ>

      <integer : num_subgrids>
      FOR subgrid = 0 TO <num_subgrids> - 1
      BEGIN
         <integer : ix>  <integer : iy>  <integer : iz>
         <integer : nx>  <integer : ny>  <integer : nz>
         <integer : rx>  <integer : ry>  <integer : rz>
            FOR j = iy TO iy + <ny> - 1
            BEGIN
               FOR i = ix TO ix + <nx> - 1
               BEGIN
                  eflx_lh_tot_ij
      	    eflx_lwrad_out_ij
      	    eflx_sh_tot_ij
      	    eflx_soil_grnd_ij
      	    qflx_evap_tot_ij
      	    qflx_evap_grnd_ij
      	    qflx_evap_soi_ij
      	    qflx_evap_veg_ij
             qflx_tran_veg_ij
      	    qflx_infl_ij
      	    swe_out_ij
      	    t_grnd_ij
           IF (clm_irr_type == 1)  qflx_qirr_ij 
      ELSE IF (clm_irr_type == 3)  qflx_qirr_inst_ij
      ELSE                         NULL
      	    FOR k = 1 TO clm_nz
      	    tsoil_ijk
      	    END
               END
            END
      END

.. _ParFlow Scattered Binary Files (.pfsb):

ParFlow Scattered Binary Files (.pfsb)
--------------------------------------

The ``.pfsb`` file format is a binary file format which is used to 
store ParFlow grid data. This format is used when the grid data 
is “scattered”, that is, when most of the data is 0. For data of 
this type, the ``.pfsb`` file format can reduce storage requirements 
considerably. The format for the file is:

.. container:: list

   ::

      <double : X>    <double : Y>    <double : Z>
      <integer : NX>  <integer : NY>  <integer : NZ>
      <double : DX>   <double : DY>   <double : DZ>

      <integer : num_subgrids>
      FOR subgrid = 0 TO <num_subgrids> - 1
      BEGIN
         <integer : ix>  <integer : iy>  <integer : iz>
         <integer : nx>  <integer : ny>  <integer : nz>
         <integer : rx>  <integer : ry>  <integer : rz>
         <integer : num_nonzero_data>
         FOR k = iz TO iz + <nz> - 1
         BEGIN
            FOR j = iy TO iy + <ny> - 1
            BEGIN
               FOR i = ix TO ix + <nx> - 1
               BEGIN
                  IF (<data_ijk> > tolerance)
                  BEGIN
                     <integer : i>  <integer : j>  <integer : k>
                     <double : data_ijk>
                  END
               END
            END
         END
      END

.. _ParFlow Solid Files (.pfsol):

ParFlow Solid Files (.pfsol)
----------------------------

The ``.pfsol`` file format is an ASCII file format which is used 
to define 3D solids. The solids are represented by closed 
triangulated surfaces, and surface “patches” may be associated 
with each solid.

Note that unlike the user input files, the solid file cannot contain
comment lines.

The format for the file is:

.. container:: list

   ::

      <integer : file_version_number>

      <integer : num_vertices>
      # Vertices
      FOR vertex = 0 TO <num_vertices> - 1
      BEGIN
         <real : x>  <real : y>  <real : z>
      END

      # Solids
      <integer : num_solids>
      FOR solid = 0 TO <num_solids> - 1
      BEGIN
         #Triangles
         <integer : num_triangles>
         FOR triangle = 0 TO <num_triangles> - 1
         BEGIN
            <integer : v0> <integer : v1> <integer : v2>
         END

         # Patches
         <integer : num_patches>
         FOR patch = 0 TO <num_patches> - 1
         BEGIN
            <integer : num_patch_triangles>
            FOR patch_triangle = 0 TO <num_patch_triangles> - 1
            BEGIN
               <integer : t>
            END
         END
      END

The field ``<file_version_number>`` is used to make file format 
changes more manageable. The field ``<num_vertices>`` specifies 
the number of vertices to follow. The fields ``<x>``, ``<y>``, 
and ``<z>`` define the coordinate of a triangle vertex. The 
field ``<num_solids>`` specifies the number of solids to follow. 
The field ``<num_triangles>`` specifies the number of triangles 
to follow. The fields ``<v0>``, ``<v1>``, and ``<v2>`` are 
vertex indexes that specify the 3 vertices of a triangle. 
Note that the vertices for each triangle MUST be specified in 
an order that makes the normal vector point outward from the 
domain. The field ``<num_patches>`` specifies the number of 
surface patches to follow. The field ``num_patch_triangles`` 
specifies the number of triangles indices to follow (these 
triangles make up the surface patch). The field ``<t>`` is 
an index of a triangle on the solid ``solid``.

ParFlow ``.pfsol`` files can be created from GMS ``.sol`` files 
using the utility ``gmssol2pfsol`` located in the ``$PARFLOW_DIR/bin`` 
directory. This conversion routine takes any number of 
GMS ``.sol`` files, concatenates the vertices of the solids defined 
in the files, throws away duplicate vertices, then prints out 
the ``.pfsol`` file. Information relating the solid index in the 
resulting ``.pfsol`` file with the GMS names and material IDs are 
printed to stdout.

.. _ParFlow Well Output File (.wells):

ParFlow Well Output File (.wells)
---------------------------------

A well output file is produced by ParFlow when wells are defined. The
well output file contains information about the well data being used in
the internal computations and accumulated statistics about the
functioning of the wells.

The header section has the following format:

.. container:: list

   ::

      LINE
      BEGIN
         <real : BackgroundX>
         <real : BackgroundY>
         <real : BackgroundZ>
         <integer : BackgroundNX>
         <integer : BackgroundNY>
         <integer : BackgroundNZ>
         <real : BackgroundDX>
         <real : BackgroundDY>
         <real : BackgroundDZ>
      END

      LINE
      BEGIN
         <integer : number_of_phases>
         <integer : number_of_components>
         <integer : number_of_wells>
      END

      FOR well = 0 TO <number_of_wells> - 1
      BEGIN
         LINE
         BEGIN
            <integer : sequence_number>
         END

         LINE
         BEGIN
            <string : well_name>
         END

         LINE
         BEGIN
            <real : well_x_lower>
            <real : well_y_lower>
            <real : well_z_lower>
            <real : well_x_upper>
            <real : well_y_upper>
            <real : well_z_upper>
            <real : well_diameter>
         END

         LINE
         BEGIN
           <integer : well_type>
           <integer : well_action>
         END
      END

The data section has the following format:

.. container:: list

   ::

      FOR time = 1 TO <number_of_time_intervals>
      BEGIN
         LINE
         BEGIN
            <real : time>
         END

         FOR well = 0 TO <number_of_wells> - 1
         BEGIN
            LINE
            BEGIN
               <integer : sequence_number>
            END

            LINE
            BEGIN
               <integer : SubgridIX>
               <integer : SubgridIY>
               <integer : SubgridIZ>
               <integer : SubgridNX>
               <integer : SubgridNY>
               <integer : SubgridNZ>
               <integer : SubgridRX>
               <integer : SubgridRY>
               <integer : SubgridRZ>
            END

            FOR well = 0 TO <number_of_wells> - 1
            BEGIN
               LINE
               BEGIN
                  FOR phase = 0 TO <number_of_phases> - 1
                  BEGIN
                     <real : phase_value>
                  END
               END

               IF injection well
               BEGIN
                  LINE
                  BEGIN
                     FOR phase = 0 TO <number_of_phases> - 1
                     BEGIN
                        <real : saturation_value>
                     END
                  END

                  LINE
                  BEGIN
                     FOR phase = 0 TO <number_of_phases> - 1
                     BEGIN
                        FOR component = 0 TO <number_of_components> - 1
                        BEGIN
                           <real : component_value>
                        END
                     END
                  END
               END

               LINE
               BEGIN
                  FOR phase = 0 TO <number_of_phases> - 1
                  BEGIN
                     FOR component = 0 TO <number_of_components> - 1
                     BEGIN
                        <real : component_fraction>
                     END
                  END
               END

               LINE
               BEGIN
                  FOR phase = 0 TO <number_of_phases> - 1
                  BEGIN
                     <real : phase_statistic>
                  END
               END

               LINE
               BEGIN
                  FOR phase = 0 TO <number_of_phases> - 1
                  BEGIN
                     <real : saturation_statistic>
                  END
               END

               LINE
               BEGIN
                  FOR phase = 0 TO <number_of_phases> - 1
                  BEGIN
                     FOR component = 0 TO <number_of_components> - 1
                     BEGIN
                        <real : component_statistic>
                     END
                  END
               END

               LINE
               BEGIN
                  FOR phase = 0 TO <number_of_phases> - 1
                  BEGIN
                     FOR component = 0 TO <number_of_components> - 1
                     BEGIN
                        <real : concentration_data>
                     END
                  END
               END
            END
         END
      END

.. _ParFlow Simple ASCII Files (.sa and .sb):

ParFlow Simple ASCII and Simple Binary Files (.sa and .sb)
----------------------------------------------------------

The simple binary, ``.sa``, file format is an ASCII file format 
which is used by ``pftools`` to write out ParFlow grid data. 
The simple binary, ``.sb``, file format is exactly the same, 
just written as BIG ENDIAN binary bit ordering :cite:p:`endian`. The format 
for the file is:

.. container:: list

   ::

      <integer : NX>  <integer : NY>  <integer : NZ>

         FOR k = 0 TO  <nz> - 1
         BEGIN
            FOR j = 0 TO  <ny> - 1
            BEGIN
               FOR i = 0 TO  <nx> - 1
               BEGIN
                  <double : data_ijk>
               END
            END
         END
