/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct indicator_data
{
   int                      num_indicators; // Number of indicators for the given solid
   int                     *indicators; //pointer to dynamic object
   char                    *filename;
   struct  indicator_data  *next_indicator_data;

   NameArray                indicator_na;
} IndicatorData;

typedef struct
{
   GeomSolid     **solids;
   int             num_solids;

   IndicatorData  *indicator_data;

   int             time_index;

   /* Geometry input names are for each "type" of geometry
      the user is inputing */
   NameArray       geom_input_names;

} PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid     *grid;
   double   *temp_data;

   /* instance data */
   Vector   *tmp_indicator_field;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * Geometries
 *--------------------------------------------------------------------------*/

void           Geometries(problem_data)
ProblemData   *problem_data;
{
   PFModule           *this_module   = ThisPFModule;
   PublicXtra         *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra       *instance_xtra = PFModuleInstanceXtra(this_module);
	            
   GeomSolid         **solids              = (public_xtra -> solids);
   int                 num_solids          = (public_xtra -> num_solids);
   IndicatorData      *indicator_data      = (public_xtra -> indicator_data);

   Grid               *grid                = (instance_xtra -> grid);
   Vector             *tmp_indicator_field = (instance_xtra -> tmp_indicator_field);
	            
   GrGeomSolid       **gr_solids;

   GrGeomExtentArray  *extent_array;

   IndicatorData      *current_indicator_data;

   CommHandle         *handle;

   int                 i, k;

   BeginTiming(public_xtra -> time_index);

   gr_solids = ctalloc(GrGeomSolid *, num_solids);

   /*-----------------------------------------------------------------------
    * Convert Geom solids to GrGeom solids
    *-----------------------------------------------------------------------*/

   extent_array = GrGeomCreateExtentArray(GridSubgrids(grid), 3, 3, 3, 3, 3, 3);
   for (i = 0; i < num_solids; i++)
   {
      if (solids[i])
      {
         GrGeomSolidFromGeom(&gr_solids[i], solids[i], extent_array);
      }
   }
   GrGeomFreeExtentArray(extent_array);

   /*-----------------------------------------------------------------------
    * Convert indicator solids to GrGeom solids
    *-----------------------------------------------------------------------*/

   current_indicator_data = indicator_data;

   i = 0;
   while(current_indicator_data != NULL)
   {
      InitVectorAll(tmp_indicator_field, -1.0);
      ReadPFBinary((current_indicator_data -> filename), tmp_indicator_field);
      handle = InitVectorUpdate(tmp_indicator_field, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      for (k = 0; k < (current_indicator_data -> num_indicators); k++)
      {
	 while (gr_solids[i])
         {
            i++;
         }
         GrGeomSolidFromInd(&gr_solids[i], tmp_indicator_field, (current_indicator_data -> indicators)[k]);
      }

      current_indicator_data = (current_indicator_data -> next_indicator_data);
   }

   EndTiming(public_xtra -> time_index);

   ProblemDataNumSolids(problem_data) = num_solids;
   ProblemDataSolids(problem_data)    = solids;   
   ProblemDataGrSolids(problem_data)  = gr_solids;

   return;
}


/*--------------------------------------------------------------------------
 * GeometriesInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *GeometriesInitInstanceXtra(grid)
Grid      *grid;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;


   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `grid'
    *-----------------------------------------------------------------------*/

   if ( grid != NULL)
   {
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
	 FreeTempVector(instance_xtra -> tmp_indicator_field);
      }

      /* set new data */
      (instance_xtra -> grid) = grid;

      /* temp_data */
      (instance_xtra -> tmp_indicator_field) = NewTempVector(grid, 1, 1);
   }

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/
   if(instance_xtra -> temp_data == NULL) 
   {
      (instance_xtra -> temp_data) = talloc(double, SizeOfVector((instance_xtra -> tmp_indicator_field)));
      SetTempVectorData((instance_xtra -> tmp_indicator_field), (instance_xtra -> temp_data));
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}  


/*--------------------------------------------------------------------------
 * GeometriesFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  GeometriesFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if(instance_xtra)
   {
      FreeTempVector(instance_xtra -> tmp_indicator_field);

      if(instance_xtra -> temp_data)
      {
	 tfree(instance_xtra -> temp_data);
      }

      tfree(instance_xtra);
   }
}  


/*--------------------------------------------------------------------------
 * GeometriesNewPublicXtra:
 *--------------------------------------------------------------------------*/

PFModule   *GeometriesNewPublicXtra()
{
   PFModule       *this_module   = ThisPFModule;
   PublicXtra     *public_xtra;

   int             num_solids;
   GeomSolid     **solids = NULL;

   int             num_new_solids;
   GeomSolid     **new_solids;

   GeomSolid     **tmp_solids;

   IndicatorData  *indicator_data, *new_indicator_data, *current_indicator_data;

   int             i, g, intype, num_intypes;

   NameArray       switch_na;
   NameArray       geom_input_na;

   char *geom_input_names;
   char key[NA_MAX_KEY_LENGTH];

   char *intype_name;

   char *geom_name;

   char *patch_names;

   /*----------------------------------------------------------
    * The name array to map names to switch values 
    *----------------------------------------------------------*/
   switch_na = NA_NewNameArray("IndicatorField SolidFile Box");


   public_xtra = ctalloc(PublicXtra, 1);

   /*----------------------------------------------------------
    * Read in the number of input types (intype).
    *----------------------------------------------------------*/

   geom_input_names = GetString("GeomInput.Names");

   geom_input_na = 
      public_xtra -> geom_input_names = NA_NewNameArray(geom_input_names);

   num_intypes = NA_Sizeof(geom_input_na);

   /*----------------------------------------------------------
    * Read in the input type, then read in the associated data.
    *----------------------------------------------------------*/

   num_solids = 0;
   indicator_data = NULL;

   for (i = 0; i < num_intypes; i++)
   {

      sprintf(key, "GeomInput.%s.InputType", 
	      NA_IndexToName(geom_input_na,i));
      intype_name = GetString(key);

      num_new_solids = 0;

      if( (intype = NA_NameToIndex(switch_na, intype_name)) < 0 )
      {
	 InputError("Error: Geometry input type <%s> is not invalid for key <%s>",
		     intype_name, key);
      }
      
      switch(intype)
      {
	 case 0: /* indicator field */
	 {
	    char *indicator_names;
	    int solids_index;

	    /* create a new indicator data structure */
	    new_indicator_data = ctalloc(IndicatorData, 1);

	    sprintf(key, "GeomInput.%s.GeomNames", 
		    NA_IndexToName(geom_input_na,i));
	    indicator_names = GetString(key);

	    
	    (new_indicator_data -> indicator_na) = 
	       NA_NewNameArray(indicator_names);

	    /* Add this geom input geometries to the total array */
	    if(GlobalsGeomNames)
	       NA_AppendToArray(GlobalsGeomNames, indicator_names);
	    else
	       GlobalsGeomNames = NA_NewNameArray(indicator_names);

	    num_new_solids = NA_Sizeof(new_indicator_data -> indicator_na);

	    (new_indicator_data -> num_indicators) = num_new_solids;

	    /* allocate and read in the indicator values */
	    (new_indicator_data -> indicators) = ctalloc(int, num_new_solids);
	    
	    for(solids_index = 0; solids_index < num_new_solids; solids_index++)
	    {
 	       sprintf(key, "GeomInput.%s.Value", 
		       NA_IndexToName(new_indicator_data -> indicator_na,
				      solids_index));
	       new_indicator_data -> indicators[solids_index] = GetInt(key);
	    }

	    /* read in the number of characters in the filename for the indicator field */

	    sprintf(key, "Geom.%s.FileName", NA_IndexToName(geom_input_na, i));
	    new_indicator_data -> filename = GetString(key);
	    
	    /* make sure and NULL out the next_indicator_data pointer */
	    (new_indicator_data -> next_indicator_data) = NULL;
	    
	    /* do some bookkeeping on the list */
	    if (indicator_data == NULL)
	    {
	       indicator_data = new_indicator_data;
	    }
	    else
	    {
	       (current_indicator_data -> next_indicator_data) = new_indicator_data;
	    }
	    current_indicator_data = new_indicator_data;
	    
	    /* allocate some new solids */
	    new_solids = ctalloc(GeomSolid *, num_new_solids);
	    
	    break;
	 }
	 
	 case 1: /* `.pfsol' file */
	 {
	    num_new_solids = GeomReadSolids(&new_solids, 
					    NA_IndexToName(geom_input_na,i),
					    GeomTSolidType);

	    sprintf(key, "GeomInput.%s.GeomNames", 
		    NA_IndexToName(geom_input_na,i));
	    geom_name = GetString(key);

	    /* Add this geom input geometries to the total array */
	    if(GlobalsGeomNames)
	       NA_AppendToArray(GlobalsGeomNames, geom_name);
	    else
	       GlobalsGeomNames = NA_NewNameArray(geom_name);

	    /* Get the patch names; patches exist only on the first 
	     * object in the input file */
	    sprintf(key, "Geom.%s.Patches", 
		    NA_IndexToName(GlobalsGeomNames, num_solids));
	    patch_names = GetString(key);
	    new_solids[0] -> patches = NA_NewNameArray(patch_names);
	    
	    break;
	 }
	 
	 case 2: /* box */
	 {
	    double        xl, yl, zl, xu, yu, zu;
	    
	    sprintf(key, "GeomInput.%s.GeomName", 
		    NA_IndexToName(geom_input_na,i));
	    geom_name = GetString(key);

	    /* Add this geom input geometries to the total array */
	    if(GlobalsGeomNames)
	       NA_AppendToArray(GlobalsGeomNames, geom_name);
	    else
	       GlobalsGeomNames = NA_NewNameArray(geom_name);

	    /* Input the box extents */	    
 	    sprintf(key, "Geom.%s.Lower.X", geom_name);
	    xl = GetDouble(key);
 	    sprintf(key, "Geom.%s.Lower.Y", geom_name);
	    yl = GetDouble(key);
 	    sprintf(key, "Geom.%s.Lower.Z", geom_name);
	    zl = GetDouble(key);

 	    sprintf(key, "Geom.%s.Upper.X", geom_name);
	    xu = GetDouble(key);
 	    sprintf(key, "Geom.%s.Upper.Y", geom_name);
	    yu = GetDouble(key);
 	    sprintf(key, "Geom.%s.Upper.Z", geom_name);
	    zu = GetDouble(key);
	    
	    new_solids = ctalloc(GeomSolid *, 1);
	    new_solids[0] =
	       GeomSolidFromBox(xl, yl, zl, xu, yu, zu, GeomTSolidType);
	    num_new_solids = 1;

	    /* Get the patch names */
	    sprintf(key, "Geom.%s.Patches", geom_name);
	    patch_names = GetStringDefault(key, 
					   "left right front back bottom top");
	    new_solids[0] -> patches = NA_NewNameArray(patch_names);
	    
	    break;
	 }

   


      }

      /*-------------------------------------------------------
       * Add the new_solids to the solids list.
       *-------------------------------------------------------*/

      if (num_new_solids)
      {
	 tmp_solids = solids;
	 solids = ctalloc(GeomSolid *, (num_solids + num_new_solids));
	 
	 for (g = 0; g < num_solids; g++)
	    solids[g] = tmp_solids[g];
	 for (g = 0; g < num_new_solids; g++)
	    solids[num_solids + g] = new_solids[g];
	 
	 tfree(tmp_solids);
	 tfree(new_solids);
	 
	 num_solids += num_new_solids;
      }
   }

   (public_xtra -> solids)         = solids;
   (public_xtra -> num_solids)     = num_solids;
   (public_xtra -> indicator_data) = indicator_data;

   (public_xtra -> time_index) = RegisterTiming("Geometries");

   /* Geometries need to be world accessible */
   GlobalsGeometries = solids;
   
   NA_FreeNameArray(switch_na);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * GeometriesFreePublicXtra
 *--------------------------------------------------------------------------*/

void  GeometriesFreePublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   IndicatorData *current_indicator_data, *tmp_indicator_data;

   int            g;

   if(public_xtra)
   {

      NA_FreeNameArray(public_xtra -> geom_input_names);

      NA_FreeNameArray(GlobalsGeomNames);

      /* SGS memory leak here */
      for (g = 0; g < (public_xtra -> num_solids); g++)
         if (public_xtra -> solids[g])
            GeomFreeSolid(public_xtra -> solids[g]);
      tfree(public_xtra -> solids);


      current_indicator_data = (public_xtra -> indicator_data);
      while(current_indicator_data != NULL)
      {
         tmp_indicator_data = (current_indicator_data -> next_indicator_data);

         /*tfree((current_indicator_data -> indicators));
         tfree((current_indicator_data -> filename));
         tfree(current_indicator_data);*/

         current_indicator_data = tmp_indicator_data;
      }

      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * GeometriesSizeOfTempData
 *--------------------------------------------------------------------------*/

int  GeometriesSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   int  sz = 0;


   /* add local TempData size to `sz' */
   sz += SizeOfVector(instance_xtra -> tmp_indicator_field);

   return sz;
}

