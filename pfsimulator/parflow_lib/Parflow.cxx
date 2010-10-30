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

#include "Parflow.hxx"

#ifdef HAVE_SAMRAI

#include "SAMRAI/hier/VariableDatabase.h"
#include "SAMRAI/pdat/CellVariable.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"
#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/RealMappedBoxConstIterator.h"
#include "SAMRAI/pdat/CellData.h"
#include "SAMRAI/pdat/CellIndex.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"
#include "SAMRAI/mesh/BergerRigoutsos.h"
#include "SAMRAI/mesh/TreeLoadBalancer.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/tbox/MemoryDatabase.h"

#include "parflow.h"

using namespace SAMRAI;

const tbox::Dimension Parflow::d_dim[Parflow::number_of_grid_types] = { 
   tbox::Dimension::getInvalidDimension(), 
   tbox::Dimension(3), 
   tbox::Dimension(3),
   tbox::Dimension(3),
   tbox::Dimension(3)};

const ParflowGridType Parflow::grid_types[number_of_grid_types] = {
   invalid_grid_type,
   flow_3D_grid_type,
   surface_2D_grid_type,
   met_2D_grid_type,
   vector_clm_topsoil_grid_type,
};


const std::string Parflow::grid_type_names[number_of_grid_types] = {
   "Invalid",
   "Flow3D",
   "Surface2D",
   "Met2D",
   "CLMTopSoil"
};


const std::string Parflow::VARIABLE_NAME = "variable";

const std::string Parflow::CURRENT_CONTEXT = "Current";
const std::string Parflow::SCRATCH_CONTEXT = "Scratch";

Parflow::Parflow(
      const std::string& object_name,
      tbox::Pointer<tbox::Database> input_db) :
   d_object_name(object_name),
   d_input_db(input_db)
{

}

Parflow:: ~Parflow() 
{
}

void Parflow::initializeLevelData(
   const tbox::Pointer< hier::BasePatchHierarchy > hierarchy,
   const int level_number,
   const double init_data_time,
   const bool can_be_refined,
   const bool initial_time,
   const tbox::Pointer< hier::BasePatchLevel > old_level,
   const bool allocate_data)
{
   NULL_USE(hierarchy);
   NULL_USE(level_number);
   NULL_USE(init_data_time);
   NULL_USE(can_be_refined);
   NULL_USE(initial_time);
   NULL_USE(old_level);
   NULL_USE(allocate_data);
}

void Parflow::resetHierarchyConfiguration(
   const tbox::Pointer< hier::BasePatchHierarchy > hierarchy,
   const int coarsest_level,
   const int finest_level)
{
   NULL_USE(hierarchy);
   NULL_USE(coarsest_level);
   NULL_USE(finest_level);

}

void Parflow::advanceHierarchy(
   const tbox::Pointer< hier::BasePatchHierarchy > hierarchy,
   const double loop_time, 
   const double dt) 
{
   NULL_USE(hierarchy);
   NULL_USE(loop_time);
   NULL_USE(dt);
}


void Parflow::applyGradientDetector(
   const tbox::Pointer< hier::BasePatchHierarchy > hierarchy,
   const int level_number,
   const double error_data_time,
   const int tag_index,
   const bool initial_time,
   const bool uses_richardson_extrapolation_too)
{
   NULL_USE(hierarchy);
   NULL_USE(level_number);
   NULL_USE(error_data_time);
   NULL_USE(tag_index);
   NULL_USE(initial_time);
   NULL_USE(uses_richardson_extrapolation_too);

}

tbox::Pointer<hier::PatchHierarchy > Parflow::getPatchHierarchy(ParflowGridType grid_type) const
{
   return d_patch_hierarchy[grid_type];
}

tbox::Pointer<mesh::GriddingAlgorithm > Parflow::getGriddingAlgorithm(ParflowGridType grid_type) const
{
   return d_patch_hierarchy[grid_type];
}

tbox::Array<int> Parflow::getTagBufferArray(ParflowGridType grid_type) const
{
   return d_tag_buffer_array[grid_type];
}

tbox::Pointer<tbox::Database> Parflow::setupGridGeometryDatabase(
   ParflowGridType grid_type, 
   std::string name,       
   SAMRAI::tbox::Pointer<SAMRAI::hier::MappedBoxLevel> mapped_box_level) 
{
   
   tbox::Pointer<tbox::Database> input_db(new tbox::MemoryDatabase(name));

   Background  *bg = GlobalsBackground;

   int lower[3];
   lower[0] = BackgroundIX(bg);
   lower[1] = BackgroundIY(bg);
   lower[2] = BackgroundIZ(bg);

   double x_lo[3];
   x_lo[0] = BackgroundXLower(bg);
   x_lo[1] = BackgroundYLower(bg);
   x_lo[2] = BackgroundZLower(bg);

   int upper[3];
   upper[0] = lower[0] +  BackgroundNX(bg) - 1;
   upper[1] = lower[1] +  BackgroundNY(bg) - 1;

   double x_up[3];
   x_up[0] = BackgroundXLower(bg) + BackgroundNX(bg) * BackgroundDX(bg);
   x_up[1] = BackgroundYLower(bg) + BackgroundNY(bg) * BackgroundDY(bg);

   switch(grid_type) 
   {
      case invalid_grid_type :
      {
	 // SGS FIXME Should error here
	 break;
      }
      case flow_3D_grid_type :
      {
	 upper[2] = lower[2] +  BackgroundNZ(bg) - 1;
	 x_up[2]  = BackgroundZLower(bg) + BackgroundNZ(bg) * BackgroundDZ(bg);
	 break;
      }
      case surface_2D_grid_type :
      {
	 lower[2] = 0;
	 upper[2] = lower[2] + 1 - 1;
	 x_up[2]  = BackgroundZLower(bg) + 1 * BackgroundDZ(bg);
	 break;
      }
      case met_2D_grid_type :
      {
	 /*
	  * SGS this is ugly grabbing the value here 
	  */
	 char key[2048];
	 sprintf(key, "Solver.CLM.MetFileNT");
	 int clm_metnt = GetIntDefault(key, 1);
	 lower[2] = 0;
	 upper[2] = lower[2] + clm_metnt - 1;
	 x_up[2]  = BackgroundZLower(bg) + clm_metnt * BackgroundDZ(bg);
	 break;
      }
      case vector_clm_topsoil_grid_type :
      {
	 lower[2] = 0;
	 upper[2] = lower[2] + 10 - 1;
	 x_up[2]  = BackgroundZLower(bg) + 10 * BackgroundDZ(bg);
	 break;
      }
   }

#if 0
   tbox::DatabaseBox box(d_dim[grid_type], lower, upper);

   tbox::Array<tbox::DatabaseBox> domain(1);
   domain[0] = box;
#else

   tbox::Array<tbox::DatabaseBox> domain(mapped_box_level -> getGlobalNumberOfBoxes());

   int i = 0;
   for (hier::RealMappedBoxConstIterator mapped_box_iterator(mapped_box_level -> getGlobalizedVersion().getGlobalMappedBoxes());
           mapped_box_iterator.isValid();
	++mapped_box_iterator, ++i) {
      domain[i] = mapped_box_iterator ->getBox();
   }
#endif
  
   input_db -> putDatabaseBoxArray("domain_boxes", domain);

   input_db -> putDoubleArray("x_lo", x_lo, d_dim[grid_type].getValue());
   input_db -> putDoubleArray("x_up", x_up, d_dim[grid_type].getValue());

   return input_db;
}

void Parflow::initializePatchHierarchy(double time)
{
   NULL_USE(time);

   for(int grid_type_index = 1; grid_type_index < number_of_grid_types; ++grid_type_index) {
      ParflowGridType grid_type = grid_types[grid_type_index];

      /* Create boxes */
      tbox::Pointer< hier::MappedBoxLevel > mapped_box_level(createMappedBoxLevelFromParflowGrid(grid_type));

      std::string grid_geometry_name("CartesianGeometry" + grid_type_names[grid_type]);
      
      tbox::Pointer<geom::CartesianGridGeometry > grid_geometry(
	 new geom::CartesianGridGeometry(
	    d_dim[grid_type],
	    grid_geometry_name,
	    setupGridGeometryDatabase(grid_type, grid_geometry_name, mapped_box_level)));
      
      std::string patch_hierarchy_name("PatchHierarchy" + grid_type_names[grid_type]);
      d_patch_hierarchy[grid_type] = new hier::PatchHierarchy(patch_hierarchy_name,
						      grid_geometry);
      
      hier::VariableDatabase *variable_database(
	 hier::VariableDatabase::getDatabase());
      
      int depth = 1;
      
      tbox::Pointer< pdat::CellVariable<double> > cell_state(
	 new pdat::CellVariable<double>(d_dim[grid_type], VARIABLE_NAME, depth));
      
      tbox::Pointer< hier::VariableContext > current_context(
	 variable_database -> getContext(CURRENT_CONTEXT));
      
      tbox::Pointer< hier::VariableContext > scratch_context(
	 variable_database -> getContext(SCRATCH_CONTEXT));
      
      hier::IntVector ghosts(d_dim[grid_type], 1);
      
      getFromInput(d_input_db, false);
      
      std::string standard_tag_and_initialize_name("StandardTagAndInitialize" + grid_type_names[grid_type]);
      tbox::Pointer< mesh::StandardTagAndInitialize > standard_tag_and_initialize(
	 new mesh::StandardTagAndInitialize(
	    d_dim[grid_type],
	    standard_tag_and_initialize_name, 
	    this,
	    d_input_db -> getDatabase(standard_tag_and_initialize_name)));
      
      tbox::Pointer< mesh::BergerRigoutsos > box_generator( 
	 new mesh::BergerRigoutsos(d_dim[grid_type]));
      
      std::string load_balancer_name("LoadBalancer" + grid_type_names[grid_type]);
      tbox::Pointer< mesh::LoadBalanceStrategy > load_balancer(
	 new mesh::TreeLoadBalancer(d_dim[grid_type],
				    load_balancer_name,
				    d_input_db -> getDatabase(load_balancer_name)));
      
      std::string gridding_algorithm_name("GriddingAlgorithm" + grid_type_names[grid_type]);
      d_gridding_algorithm[grid_type] = 
	 new mesh::GriddingAlgorithm(
#if 0
	    d_dim[grid_type],
#endif
	    d_patch_hierarchy[grid_type],
	    gridding_algorithm_name,
	    d_input_db -> getDatabase(gridding_algorithm_name),
	    standard_tag_and_initialize,
	    box_generator,
	    load_balancer);

      const int d_max_levels = d_patch_hierarchy[grid_type] -> getMaxNumberOfLevels();
      
      d_tag_buffer_array[grid_type].resizeArray(d_max_levels);
      for (int il = 0; il < d_max_levels; il++) {
	 d_tag_buffer_array[grid_type][il] = 1;
      }

      const hier::IntVector one_vector(d_dim[grid_type], 1);
#if 0
      const hier::IntVector ratio(d_max_levels > 1 ? 
				  d_gridding_algorithm[grid_type] -> getRatioToCoarserLevel(1) : one_vector);
#endif
      const hier::IntVector ratio(d_max_levels > 1 ? 
				  d_patch_hierarchy[grid_type] -> getRatioToCoarserLevel(1) : one_vector);



      std::vector<hier::IntVector> fine_connector_gcw(d_max_levels,
						      hier::IntVector(d_dim[grid_type]));
      std::vector<hier::IntVector> peer_connector_gcw(d_max_levels,
						      hier::IntVector(d_dim[grid_type]));

#if 0
      d_gridding_algorithm[grid_type] -> computeAllConnectorWidths(fine_connector_gcw,
								   peer_connector_gcw,
								   *d_patch_hierarchy[grid_type] );
      
      d_patch_hierarchy[grid_type]->getMappedBoxHierarchy().setMappedBoxLevelParameters(
	 0,
	 ratio,
	 fine_connector_gcw[0],
	 peer_connector_gcw[0]);
#endif
      


      
      d_patch_hierarchy[grid_type] -> makeNewPatchLevel(0, *mapped_box_level);

      tbox::Pointer<hier::PatchLevel> level(d_patch_hierarchy[grid_type] -> getPatchLevel(0));

      for(int i = 4; i > 0; --i) {
	 level -> getMappedBoxLevel() -> getPersistentOverlapConnectors().
	    findOrCreateConnector(
	       *level -> getMappedBoxLevel(),
	       hier::IntVector(d_dim[grid_type], i));
      }
   }
}

void Parflow::getFromInput(
   tbox::Pointer<tbox::Database> db,
   bool is_from_restart) 
{
   NULL_USE(db);
   
   if (!is_from_restart) {

   }
}

tbox::Pointer< hier::MappedBoxLevel > Parflow::createMappedBoxLevelFromParflowGrid(ParflowGridType grid_type)
{
   Grid *grid = CreateGrid(GlobalsUserGrid);

   hier::MappedBoxSet mapped_box_set;

   int i;
   ForSubgridI(i, GridSubgrids(grid))
   {

      // Build a box based off of Parflow grid
      hier::Index lower(d_dim[grid_type]);
      lower[0] = SubgridIX(GridSubgrid(grid, i));
      lower[1] = SubgridIY(GridSubgrid(grid, i));
      
      
      hier::Index upper(d_dim[grid_type]);
      upper[0] = lower[0] + SubgridNX(GridSubgrid(grid, i)) - 1;
      upper[1] = lower[1] + SubgridNY(GridSubgrid(grid, i)) - 1;
     
      switch(grid_type) 
      {
	 case invalid_grid_type :
	    // SGS FIXME Should error here
	    break;
	 case flow_3D_grid_type :
	 {
	    lower[2] = SubgridIZ(GridSubgrid(grid, i));
	    upper[2] = lower[2] + SubgridNZ(GridSubgrid(grid, i)) - 1;
	    break;
	 }
	 case surface_2D_grid_type :
	 {
	    Background  *bg = GlobalsBackground;
	    lower[2] = 0;
	    upper[2] = lower[2];
	    break;
	 }
	 case met_2D_grid_type :
	 {
	    /*
	     * SGS this is ugly grabbing the value here 
	     */
	    char key[2048];
	    sprintf(key, "Solver.CLM.MetFileNT");
	    int clm_metnt = GetIntDefault(key, 1);
	    Background  *bg = GlobalsBackground;
	    lower[2] = 0;
	    upper[2] = lower[2] + clm_metnt - 1;
	    break;
	 }
	 case vector_clm_topsoil_grid_type :
	 {
	    lower[2] = 0;
	    upper[2] = lower[2] + 10 - 1;
	    break;
	 }
      }
      
      hier::Box box(lower, upper);
   
      // Build a mapped box and insert into layer.  

      const int my_rank = amps_Rank();
      
      hier::MappedBox mapped_box( box, i, my_rank );
      
      
      mapped_box_set.insert(mapped_box_set.end(), mapped_box);

   }

   const hier::IntVector ratio(d_dim[flow_3D_grid_type], 1);

   tbox::Pointer<hier::MappedBoxLevel> mapped_box_level(new hier::MappedBoxLevel(mapped_box_set, ratio));
   
   mapped_box_level -> getGlobalizedVersion();

   FreeGrid(grid);

   return mapped_box_level;
}

#endif




