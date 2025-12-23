/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

#ifndef included_parflow_Parflow
#define included_parflow_Parflow

#include "parflow_config.h"

enum ParflowGridType { 
      invalid_grid_type,
      flow_3D_grid_type,
      surface_2D_grid_type,
      met_2D_grid_type,
      vector_clm_topsoil_grid_type
};


#ifdef HAVE_SAMRAI
#include "SAMRAI/SAMRAI_config.h"

#include "SAMRAI/mesh/StandardTagAndInitStrategy.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/hier/BasePatchHierarchy.h"
#include "SAMRAI/xfer/RefineAlgorithm.h"
#include "SAMRAI/xfer/CoarsenAlgorithm.h"
#include "SAMRAI/tbox/Pointer.h"
#include "SAMRAI/tbox/Database.h"

#include <string>

class Parflow:
   public SAMRAI::mesh::StandardTagAndInitStrategy
{

public:

   static const int number_of_grid_types = 5;

   static const ParflowGridType grid_types[number_of_grid_types];
   static const std::string grid_type_names[number_of_grid_types];

   Parflow(
      const std::string& object_name,
      SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);
   
   virtual ~Parflow();

   virtual void initializeLevelData(
      const SAMRAI::tbox::Pointer< SAMRAI::hier::BasePatchHierarchy > hierarchy,
      const int level_number,
      const double init_data_time,
      const bool can_be_refined,
      const bool initial_time,
      const SAMRAI::tbox::Pointer< SAMRAI::hier::BasePatchLevel > old_level = 
      SAMRAI::tbox::Pointer< SAMRAI::hier::BasePatchLevel >(NULL),
      const bool allocate_data = true);

   virtual void resetHierarchyConfiguration(
      const SAMRAI::tbox::Pointer< SAMRAI::hier::BasePatchHierarchy > hierarchy,
      const int coarsest_level,
      const int finest_level);

   void advanceHierarchy(
      const SAMRAI::tbox::Pointer< SAMRAI::hier::BasePatchHierarchy > hierarchy,
      const double loop_time, 
      const double dt);

   virtual void applyGradientDetector(
      const SAMRAI::tbox::Pointer< SAMRAI::hier::BasePatchHierarchy > hierarchy,
      const int level_number,
      const double error_data_time,
      const int tag_index,
      const bool initial_time,
      const bool uses_richardson_extrapolation_too);

   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy > getPatchHierarchy(ParflowGridType grid_type) const;

   SAMRAI::tbox::Pointer<SAMRAI::mesh::GriddingAlgorithm > getGriddingAlgorithm(ParflowGridType grid_type) const;

   SAMRAI::tbox::Array<int> getTagBufferArray(ParflowGridType grid_type) const;

   void initializePatchHierarchy(double time);

   const SAMRAI::tbox::Dimension& getDim(ParflowGridType grid_type) const
   {
      return d_dim[grid_type];
   }

  private:

   SAMRAI::tbox::Pointer<SAMRAI::hier::MappedBoxLevel> createMappedBoxLevelFromParflowGrid(ParflowGridType grid_type);

   SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> setupGridGeometryDatabase(ParflowGridType grid_type, std::string name, 
      SAMRAI::tbox::Pointer<SAMRAI::hier::MappedBoxLevel> mapped_box_level);

   void getFromInput(
      SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db,
      bool is_from_restart);

   static const SAMRAI::tbox::Dimension d_dim[number_of_grid_types];

   // FIXME rename this
   static const std::string VARIABLE_NAME;

   static const std::string CURRENT_CONTEXT;
   static const std::string SCRATCH_CONTEXT;


   // Not implemented; standard C++'ism to avoid bad things
   Parflow(const Parflow&);
   void operator=(const Parflow&);
   
   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy > d_patch_hierarchy[number_of_grid_types];

   SAMRAI::tbox::Pointer< SAMRAI::mesh::GriddingAlgorithm > d_gridding_algorithm[number_of_grid_types];

   SAMRAI::tbox::Pointer< SAMRAI::xfer::RefineAlgorithm > d_boundary_fill_refine_algorithm[number_of_grid_types];
   SAMRAI::tbox::Pointer< SAMRAI::xfer::RefineAlgorithm > d_fill_after_regrid[number_of_grid_types];
   SAMRAI::tbox::Pointer< SAMRAI::xfer::CoarsenAlgorithm > d_coarsen_algorithm[number_of_grid_types];

   SAMRAI::tbox::Array< SAMRAI::tbox::Pointer< SAMRAI::xfer::RefineSchedule > > d_boundary_schedule_advance[number_of_grid_types];
   SAMRAI::tbox::Array< SAMRAI::tbox::Pointer< SAMRAI::xfer::CoarsenSchedule > > d_coarsen_schedule[number_of_grid_types];

   SAMRAI::tbox::Array<int> d_tag_buffer_array[number_of_grid_types];

   std::string d_object_name;

   SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> d_input_db;
};

#else

#if 0
class Parflow
{

public:

   static const int number_of_grid_types = 4;


};
#endif

#endif

#endif
