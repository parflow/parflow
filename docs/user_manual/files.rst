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

.. _CLM Driver Input Files:

CLM Driver Input Files
----------------------

When ParFlow is coupled with CLM, three driver input files configure the
land surface model. These plain-text files are read by CLM's Fortran
reader at the start of each simulation. They must be present in the
run directory.

.. _drv_clmin.dat:

``drv_clmin.dat`` — CLM Initialization Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Format:** Free-format, any order. Each non-comment line contains:

.. container:: list

   ::

      <string : parameter_name>  <value>  <description (ignored by reader)>

Comment lines start with ``!``. The reader matches the first 15
characters of each line against known parameter names; unrecognized
lines are silently skipped.

The file is read in two passes: (1) domain and classification
parameters are read into the ``drv_module`` (1-D scalars), then (2)
grid-space parameters are read into the ``grid_module`` (broadcast
uniformly across all grid cells).

**Parameter Reference:**

.. list-table:: Domain & Classification
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``maxt``
     - 1
     - Maximum tiles per grid cell (use 1 for ParFlow coupling)
   * - ``mina``
     - 0.05
     - Minimum grid area for tile (%)
   * - ``udef``
     - -9999.
     - Undefined value marker
   * - ``vclass``
     - 2
     - Vegetation classification scheme (1=UMD, 2=IGBP)

.. list-table:: File References
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``vegtf``
     - (required)
     - Vegetation tile map file (``drv_vegm.dat``)
   * - ``vegpf``
     - (required)
     - Vegetation type parameter file (``drv_vegp.dat``)
   * - ``outf1d``
     - (required)
     - CLM 1-D output filename
   * - ``poutf1d``
     - (required)
     - CLM 1-D parameter output filename
   * - ``rstf``
     - (required)
     - CLM restart file prefix

.. list-table:: Run Timing
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``startcode``
     - 2
     - Start mode: 1=restart file, 2=cold start from this file
   * - ``sss``, ``smn``, ``shr``, ``sda``, ``smo``, ``syr``
     - (required)
     - Start time (second, minute, hour, day, month, year)
   * - ``ess``, ``emn``, ``ehr``, ``eda``, ``emo``, ``eyr``
     - (required)
     - End time (second, minute, hour, day, month, year)

.. warning::

   **All times must be in UTC.** The start/end times in this file, the
   meteorological forcing data, and ParFlow's own timing must all use
   coordinated universal time (UTC). CLM computes the **local solar time**
   internally from the latitude and longitude specified in ``drv_vegm.dat``
   to determine the solar zenith angle for radiation calculations.
   If you supply forcing data in local time, the solar geometry will be
   wrong — radiation will be out of phase with the forcing, producing
   incorrect surface energy balance, snowmelt timing, and
   evapotranspiration.

   The forcing file timestamps, the ``shr``/``sda``/``smo``/``syr``
   start time here, and ParFlow's ``TimingInfo.StartTime`` must all
   refer to the same UTC instant.

.. list-table:: Initial Conditions
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``clm_ic``
     - 2
     - IC source: 1=restart file, 2=defined in this file
   * - ``t_ini``
     - 300.
     - Initial temperature [K]
   * - ``h2osno_ini``
     - 0.
     - Initial snow water equivalent [mm]
   * - ``sw_ini``
     - (optional)
     - Initial soil water fraction [0-1]

.. list-table:: Diagnostic Output
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``surfind``
     - 2
     - Number of surface diagnostic variables to output
   * - ``soilind``
     - 1
     - Number of soil layer diagnostic variables to output
   * - ``snowind``
     - 0
     - Number of snow layer diagnostic variables to output

.. list-table:: Forcing Heights
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``forc_hgt_u``
     - 10.0
     - Observational height of wind [m]
   * - ``forc_hgt_t``
     - 2.0
     - Observational height of temperature [m]
   * - ``forc_hgt_q``
     - 2.0
     - Observational height of humidity [m]

.. note::

   **Forcing heights must match the forcing data source.**
   Use 30/30/30 m for gridded reanalysis products (NLDAS, CW3E, AORC)
   which report at ~30 m AGL. Use 10/2/2 m for site-level meteorological
   stations or low-canopy simulations. Mismatched heights create
   unrealistic aerodynamic resistance, especially over tall vegetation.

.. list-table:: Vegetation
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dewmx``
     - 0.1
     - Maximum allowed dew [mm] (CLM4.5+ uses 0.2)
   * - ``rootfr``
     - -9999.0
     - Root fraction depth average (-9999 = use PFT default)

.. list-table:: Roughness Lengths
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``zlnd``
     - 0.01
     - Roughness length for soil [m]
   * - ``zsno``
     - 0.0024
     - Roughness length for snow [m]
   * - ``csoilc``
     - 0.0025
     - Drag coefficient for soil under canopy [-]

.. list-table:: Numerical Parameters
   :widths: 15 10 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``capr``
     - 0.34
     - Tuning factor: first-layer T to surface T
   * - ``cnfac``
     - 0.5
     - Crank-Nicholson factor [0-1]
   * - ``smpmin``
     - -1.0e8
     - Minimum soil matric potential [mm]
   * - ``ssi``
     - 0.033
     - Irreducible water saturation of snow
   * - ``wimp``
     - 0.05
     - Water impermeable threshold for porosity

.. admonition:: Deprecated Parameters

   The following parameters appear in legacy ``drv_clmin.dat`` files but
   have **no effect** on ParFlow-CLM simulations. They may be safely
   removed. If present, they are silently read but never used in any
   active computation path.

   - ``nt`` — Not read by the Fortran reader at all
   - ``metf1d`` — Read but unused in ParFlow-coupled mode (forcing comes via ParFlow keys)
   - ``qflx_tran_vegmx`` — Not read by the Fortran reader at all
   - ``hkdepth`` — Read into tile space; downstream formula is commented out
   - ``wtfact`` — Read into CLM; feeds ``fcov`` which is never used by ParFlow
   - ``trsmx0`` — Read into CLM; never referenced in any computation
   - ``smpmax`` — Read into CLM; only used in ALMA output (deprecated)
   - ``scalez`` — Read into tile space; downstream formula is commented out
   - ``pondmx`` — Read into CLM; feeds ``xs`` computation whose downstream code is all commented out

   A cleaned file without these parameters is provided as
   ``test/tcl/clm/drv_clmin_clean.dat``. A best-practice version with
   recommended forcing heights and canopy parameters is provided as
   ``test/tcl/clm/drv_clmin_bestpractice.dat``.

**Annotated example** (minimal working file):

.. code-block:: text

   ! --- Domain ---
   maxt           1              Maximum tiles per grid
   mina           0.05           Min grid area for tile (%)
   udef           -9999.         Undefined value
   vclass         2              IGBP vegetation classification
   !
   ! --- Files ---
   vegtf          drv_vegm.dat   Vegetation tile map
   vegpf          drv_vegp.dat   Vegetation parameters
   outf1d         clm.output.txt CLM output file
   poutf1d        clm.para.out   Parameter output file
   rstf           clm.rst.       Restart file prefix
   !
   ! --- Timing ---
   startcode      2              Cold start
   sss 00  smn 00  shr 00  sda 01  smo 10  syr 2000
   ess 00  emn 00  ehr 00  eda 01  emo 10  eyr 2001
   clm_ic         2              ICs from this file
   t_ini          285.           Initial temperature [K]
   h2osno_ini     0.             Initial SWE [mm]
   !
   ! --- Diagnostics ---
   surfind 2  soilind 1  snowind 0
   !
   ! --- Forcing heights (30m for gridded products) ---
   forc_hgt_u     30.0           Wind height [m]
   forc_hgt_t     30.0           Temperature height [m]
   forc_hgt_q     30.0           Humidity height [m]
   !
   ! --- Vegetation ---
   dewmx          0.2            Max dew [mm]
   rootfr         -9999.0        Use PFT default
   !
   ! --- Roughness ---
   zlnd 0.01  zsno 0.0024  csoilc 0.0025
   !
   ! --- Numerical ---
   capr 0.34  cnfac 0.5  smpmin -1.e8  ssi 0.033  wimp 0.05


.. _drv_vegp.dat:

``drv_vegp.dat`` — Vegetation Type Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Format:** Keyword-value blocks. Each block has a keyword line followed
by a data line containing one value per vegetation class (18 values for
IGBP). Comment lines start with ``!``. Blocks may appear in any order.

.. code-block:: text

   <keyword>       <description (ignored)>
   <val_1> <val_2> ... <val_18>

The reader matches the first 15 characters of the keyword line against
known parameter names. A value of ``-99.`` means "not applicable" (used
for snow/ice, water, bare soil classes where a parameter is meaningless).

**IGBP Land Cover Classes:**

.. list-table::
   :widths: 5 40
   :header-rows: 1

   * - Class
     - Description
   * - 1
     - Evergreen needleleaf forests
   * - 2
     - Evergreen broadleaf forests
   * - 3
     - Deciduous needleleaf forests
   * - 4
     - Deciduous broadleaf forests
   * - 5
     - Mixed forests
   * - 6
     - Closed shrublands
   * - 7
     - Open shrublands
   * - 8
     - Woody savannas
   * - 9
     - Savannas
   * - 10
     - Grasslands
   * - 11
     - Permanent wetlands
   * - 12
     - Croplands
   * - 13
     - Urban and built-up lands
   * - 14
     - Cropland/natural vegetation mosaics
   * - 15
     - Snow and ice
   * - 16
     - Barren or sparsely vegetated
   * - 17
     - Water bodies
   * - 18
     - Bare soil

**Parameter Reference:**

.. list-table:: Structural Parameters
   :widths: 15 55
   :header-rows: 1

   * - Parameter
     - Description
   * - ``itypwat``
     - Water type (1=soil, 2=land ice, 3=deep lake, 4=shallow lake, 5=wetland)
   * - ``lai``
     - Maximum leaf area index [-]
   * - ``lai0``
     - Minimum leaf area index [-]
   * - ``sai``
     - Stem area index [-]
   * - ``z0m``
     - Aerodynamic roughness length [m]
   * - ``displa``
     - Displacement height [m]
   * - ``dleaf``
     - Leaf dimension [m]
   * - ``roota``
     - Root distribution parameter a (Zeng 2001)
   * - ``rootb``
     - Root distribution parameter b (Zeng 2001)

.. list-table:: Optical Parameters
   :widths: 15 55
   :header-rows: 1

   * - Parameter
     - Description
   * - ``rhol_vis``
     - Leaf reflectance, visible band
   * - ``rhol_nir``
     - Leaf reflectance, near-infrared band
   * - ``rhos_vis``
     - Stem reflectance, visible band
   * - ``rhos_nir``
     - Stem reflectance, near-infrared band
   * - ``taul_vis``
     - Leaf transmittance, visible band
   * - ``taul_nir``
     - Leaf transmittance, near-infrared band
   * - ``taus_vis``
     - Stem transmittance, visible band
   * - ``taus_nir``
     - Stem transmittance, near-infrared band
   * - ``xl``
     - Leaf/stem orientation index (-0.4 to 0.6; 0 = spherical)

.. list-table:: Hydrology Parameters
   :widths: 15 55
   :header-rows: 1

   * - Parameter
     - Description
   * - ``vw``
     - Beta transpiration exponent: ``[(h2osoi_vol-watdry)/(watopt-watdry)]^vw``
   * - ``irrig``
     - Irrigation flag (0=none, 1=irrigate)

.. list-table:: Photosynthesis Parameters (optional)
   :widths: 15 55
   :header-rows: 1

   * - Parameter
     - Description
   * - ``vcmx25``
     - Maximum carboxylation rate at 25 C [umol CO2/m2/s] — **activates PFT photosynthesis**
   * - ``c3psn``
     - Photosynthetic pathway (1=C3, 0=C4)
   * - ``mp``
     - Ball-Berry slope parameter
   * - ``bp``
     - Minimum leaf conductance [umol/m2/s]
   * - ``qe25``
     - Quantum efficiency at 25 C [umol CO2/umol photon]
   * - ``folnmx``
     - Foliage nitrogen concentration when f(N)=1 [%]
   * - ``g1_medlyn``
     - Medlyn stomatal slope parameter [kPa^0.5]
   * - ``clump``
     - Canopy clumping index (1.0 = no clumping)

.. note::

   **PFT-specific photosynthesis is activated by the presence of
   ``vcmx25`` in the file.** Without it, CLM uses hardcoded defaults
   from ``clm_varcon.F90``. When ``vcmx25`` is present, the ``c3psn``,
   ``mp``, ``bp``, ``qe25``, ``folnmx``, and ``g1_medlyn`` values are
   used per-PFT. A best-practice file with CLM4.5 corrections and
   photosynthesis parameters is provided as
   ``test/tcl/clm/drv_vegp_bestpractice.dat``.

**CLM4.5 corrections** (applied in ``drv_vegp_bestpractice.dat``):

- IGBP 10 (grasslands): ``sai`` 4.0 → 0.5, ``roota`` 1.0 → 11.0 (Zeng 2001)
- IGBP 8-10, 12 (savanna/grass/crop): ``taus_vis/nir`` → 0.001, ``rhos_vis`` → 0.16, ``rhos_nir`` → 0.39 (CLM4.5 Table 3.1)
- IGBP 9, 10, 12: ``rhol_nir`` 0.58 → 0.35
- C3/C4 fixes: IGBP 10 ``vcmx25`` 52 → 24, ``qe25`` 0.04 → 0.05 (C4 pathway)

.. _drv_vegm.dat:

``drv_vegm.dat`` — Vegetation Tile Map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Format:** Two header lines followed by one data line per grid cell.

.. code-block:: text

   x  y  lat    lon    sand clay color  fractional coverage (18 IGBP classes)
          (Deg)  (Deg)  (%/100)  index  1    2    3    ...  18
   1  1  34.75  -98.14  0.16 0.265  2  0.0 0.0 ... 1.0 ... 0.0

**Column specification:**

.. list-table::
   :widths: 15 55
   :header-rows: 1

   * - Column
     - Description
   * - ``x``, ``y``
     - Grid cell indices (1-based)
   * - ``lat``, ``lon``
     - Latitude and longitude [degrees]. **Used by CLM to compute local solar
       time** for solar zenith angle calculations (see warning below).
   * - ``sand``
     - Sand fraction [0-1] (NOT percent)
   * - ``clay``
     - Clay fraction [0-1] (NOT percent)
   * - ``color``
     - Soil color index (1-20; controls dry/wet soil albedo)
   * - Classes 1-18
     - Fractional coverage by each IGBP class (should sum to 1.0)

**Notes:**

- The file must contain exactly ``NX × NY`` data lines (one per grid cell),
  ordered with x varying fastest.
- Sand and clay are fractional (0-1), not percent. The remainder (1 - sand - clay)
  is implicitly silt.
- The ``color`` index selects a soil albedo pair from CLM's lookup table.
  Values 1-8 are most common.
- For single-column (1×1) simulations, the file has 2 header lines + 1 data line.
- Fractional coverages must sum to 1.0. For bare soil simulations,
  set class 18 to 1.0 and all others to 0.0.

.. warning::

   **Latitude and longitude control solar geometry.** CLM uses the
   ``lat`` and ``lon`` values in this file to convert UTC time to local
   solar time for computing the solar zenith angle. This affects
   shortwave radiation partitioning, snow albedo, snowmelt timing, and
   photosynthesis. Incorrect coordinates will produce systematically
   wrong diurnal radiation cycles even if the forcing data itself is
   correct. Ensure that the coordinates here match the actual location
   of your forcing data, and that **all timestamps (forcing data,
   drv_clmin.dat start/end times, and ParFlow timing) are in UTC**.

**Single-column example** (bare soil at 34.75°N, 98.14°W):

.. code-block:: text

    x  y  lat    lon    sand clay color  fractional coverage of grid by vegetation class
          (Deg)  (Deg)  (%/100)   index  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18
    1  1  34.750 -98.138  0.16 0.265   2   0.0 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0

**Grassland example** (IGBP class 10):

.. code-block:: text

    x  y  lat    lon    sand clay color  fractional coverage of grid by vegetation class
          (Deg)  (Deg)  (%/100)   index  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18
    1  1  40.000 -105.00  0.40 0.20   4   0.0 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0


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
