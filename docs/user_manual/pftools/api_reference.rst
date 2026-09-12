PFTools API Reference
=====================

Each entry below includes a short description from the object's docstring.
Click a name to open the full documentation.

.. toctree::
   :maxdepth: 1
   :hidden:

   parflow
   run
   builders
   io
   hydrology
   fs
   export
   compare
   top
   settings
   pf_backend

.. rubric:: Parflow Module

.. currentmodule:: parflow

.. autosummary::
   :nosignatures:

   ParflowBinaryReader
   Run
   read_pfb
   write_pfb
   read_pfb_sequence
   pf_test_file
   pf_test_file_with_abs

.. rubric:: Run Class

.. currentmodule:: parflow.tools.core

.. autosummary::
   :nosignatures:

   Run.from_definition
   Run.get_name
   Run.set_name
   Run.write
   Run.write_subsurface_table
   Run.clone
   Run.run
   Run.check_nans
   Run.dist
   Run.undist
   Run.details
   Run.doc
   Run.get_children_of_type
   Run.get_context_settings
   Run.keys
   Run.pfset
   Run.select
   Run.to_dict
   Run.to_pf_name
   Run.validate
   Run.value
   Run.data_accessor
   Run.full_name
   check_parflow_execution
   get_current_parflow_version
   get_process_args
   update_run_from_args

.. rubric:: Builders

.. currentmodule:: parflow.tools.builders

.. autosummary::
   :nosignatures:

   SolidFileBuilder
   SubsurfacePropertiesBuilder
   ReservoirPropertiesBuilder
   VegParamBuilder
   DomainBuilder
   CLMImporter
   TableToProperties

.. rubric:: IO Module

.. currentmodule:: parflow.tools.io

.. autosummary::
   :nosignatures:

   ParflowBinaryReader
   DataAccessor
   read_pfb
   write_pfb
   read_pfb_sequence
   read_pfsb
   write_dist
   undist
   load_patch_matrix_from_image_file
   load_patch_matrix_from_asc_file
   load_patch_matrix_from_sa_file
   write_patch_matrix_as_asc
   write_patch_matrix_as_sa
   read_pfidb
   read_yaml
   read_clm
   write_dict
   write_dict_as_pfidb
   write_dict_as_yaml
   write_dict_as_json
   to_native_type
   get_maingrid_and_remainder
   get_subgrid_loc
   subgrid_lower_left
   subgrid_size
   precalculate_subgrid_info

.. rubric:: Hydrology Module

.. currentmodule:: parflow.tools.hydrology

.. autosummary::
   :nosignatures:

   calculate_evapotranspiration
   calculate_overland_flow
   calculate_overland_flow_grid
   calculate_overland_fluxes
   calculate_subsurface_storage
   calculate_surface_storage
   calculate_water_table_depth
   compute_hydraulic_head
   compute_water_table_depth

.. rubric:: FS Module

.. currentmodule:: parflow.tools.fs

.. autosummary::
   :nosignatures:

   cp
   get_absolute_path
   mkdir
   rm
   get_text_file_content
   exists
   chdir

.. rubric:: Export Module

.. currentmodule:: parflow.tools.export

.. autosummary::
   :nosignatures:

   SubsurfacePropertiesExporter
   CLMExporter

.. rubric:: Compare Module

.. currentmodule:: parflow.tools.compare

.. autosummary::
   :nosignatures:

   pf_test_file
   pf_test_file_with_abs
   pf_test_equal
   msig_diff

.. rubric:: Top Module

.. currentmodule:: parflow.tools.top

.. autosummary::
   :nosignatures:

   compute_top
   extract_top

.. rubric:: Settings Module

.. currentmodule:: parflow.tools.settings

.. autosummary::
   :nosignatures:

   get_working_directory
   set_working_directory
   enable_line_error
   disable_line_error
   enable_exit_error
   disable_exit_error
   set_parflow_version

.. rubric:: PF Backend Module

.. currentmodule:: parflow.tools.pf_backend

.. autosummary::
   :nosignatures:

   ParflowBackendArray
