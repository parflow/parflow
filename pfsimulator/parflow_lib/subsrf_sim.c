/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/
/*****************************************************************************
*
*****************************************************************************/

#include "parflow.h"

#include <string.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int type;                      /* Type of full perm coeff. data - input */
                                 /*   type = 0: kx, ky, kz specified const.
                                  *             in each geounit.
                                  *   type = 1: kx, ky, kz specified cell-wise
                                  *             by files */
  void      *data;               /* Ptr. to type structure */

  int nc;                        /* Number of conditioning data points */
  double     *x, *y, *z;         /* x,y,z coordinates of conditioning data */
  double     *v;                 /* Conditioning data values */
  int ctype;                     /* 0 = single pt., 1 = vertical line of pts.*/

  int num_geo_indexes;
  int        *geo_indexes;      /* Integer value of each geounit */

  PFModule **KFieldSimulators;

  int time_index;

  NameArray geo_indexes_na;
} PublicXtra;

typedef struct {
  PFModule **KFieldSimulators;

  Grid      *grid;

  double    *temp_data;

  double    *temp_data1;
} InstanceXtra;

typedef struct {
  NameArray tens_indexes_na;
  int num_tens_indexes;
  int    *tens_indexes;

  double *kx_values;
  double *ky_values;
  double *kz_values;
} Type0;

typedef struct {
  char   *kx_file_name;
  char   *ky_file_name;
  char   *kz_file_name;

  Vector *kx_values;
  Vector *ky_values;
  Vector *kz_values;
} Type1;

/*--------------------------------------------------------------------------
 * SubsrfSim
 *--------------------------------------------------------------------------*/

void SubsrfSim(
               ProblemData * problem_data,
               Vector *      perm_x,
               Vector *      perm_y,
               Vector *      perm_z,
               int           num_geounits,
               GeomSolid **  geounits,
               GrGeomSolid **gr_geounits)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Type0         *dummy0;
  Type1         *dummy1;

  int num_geo_indexes = (public_xtra->num_geo_indexes);
  int              *geo_indexes = (public_xtra->geo_indexes);

  PFModule        **KFieldSimulators = (instance_xtra->KFieldSimulators);

  RFCondData       *cdata;

  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;

  VectorUpdateCommHandle       *handle;

  GrGeomSolid      *gr_solid, *gr_domain;

  Grid             *grid = VectorGrid(perm_x);

  SubgridArray     *subgrids;

  Subgrid          *subgrid,
                   *well_subgrid,
                   *tmp_subgrid;

  Subvector        *perm_x_sub, *perm_y_sub, *perm_z_sub;
  Subvector        *kx_values_sub, *ky_values_sub, *kz_values_sub;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_p, ny_p, nz_p;
  double dx, dy, dz;
  int r;
  int i, j, k, pi, sg, well;
  int ipx, ipy, ipz;
  double           *perm_x_elt, *perm_y_elt, *perm_z_elt;

  double           *perm_x_dat, *perm_y_dat, *perm_z_dat;
  double           *kx_values_dat, *ky_values_dat, *kz_values_dat;
  double well_volume, cell_volume;
  double perm_average_x, perm_average_y, perm_average_z;

  amps_Invoice result_invoice;

  (void)num_geounits;

  BeginTiming(public_xtra->time_index);

  /* Transfer conditioning data from the public_xtra structure to a local
   * structure for ease in passing it to the random field functions. */
  cdata = ctalloc(RFCondData, 1);
  (cdata->nc) = (public_xtra->nc);
  (cdata->x) = (public_xtra->x);
  (cdata->y) = (public_xtra->y);
  (cdata->z) = (public_xtra->z);
  (cdata->v) = (public_xtra->v);

  /*------------------------------------------------------------------------
   * Compute permeability in the geounits
   *------------------------------------------------------------------------*/

  /*
   * Compute perm in z component, this will later be assigned to x,y with tensor
   * applied.
   */
  for (i = 0; i < num_geo_indexes; i++)
  {
    j = geo_indexes[i];
    PFModuleInvokeType(KFieldSimulatorInvoke,
                       KFieldSimulators[i],
                       (geounits[j], gr_geounits[j], perm_z, cdata));
  }

  /*------------------------------------------------------------------------
   * Multiply by scalars k_x, k_y and k_z which can vary by cell
   *------------------------------------------------------------------------*/



  switch (public_xtra->type)
  {
    case 0:  /* kx, ky, kz specified constants in each geounit */
    {
      double *kx_values;
      double *ky_values;
      double *kz_values;

      int ir;

      int num_tens_indexes;
      int    *tens_indexes;

      dummy0 = (Type0*)(public_xtra->data);

      tens_indexes = dummy0->tens_indexes;
      num_tens_indexes = dummy0->num_tens_indexes;

      kx_values = dummy0->kx_values;
      ky_values = dummy0->ky_values;
      kz_values = dummy0->kz_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      for (ir = 0; ir < num_tens_indexes; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, tens_indexes[ir]);

        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

          perm_x_sub = VectorSubvector(perm_x, sg);
          perm_y_sub = VectorSubvector(perm_y, sg);
          perm_z_sub = VectorSubvector(perm_z, sg);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          perm_x_dat = SubvectorData(perm_x_sub);
          perm_y_dat = SubvectorData(perm_y_sub);
          perm_z_dat = SubvectorData(perm_z_sub);


          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ipx = SubvectorEltIndex(perm_x_sub, i, j, k);
            ipy = SubvectorEltIndex(perm_y_sub, i, j, k);
            ipz = SubvectorEltIndex(perm_z_sub, i, j, k);

            perm_x_dat[ipx] = perm_z_dat[ipz] * kx_values[ir];
            perm_y_dat[ipy] = perm_z_dat[ipz] * ky_values[ir];
            perm_z_dat[ipz] = perm_z_dat[ipz] * kz_values[ir];
          });

          // SGS if this is important here why does it not appear in loop below?
          // SGS This loop should not be here, looping over outside of domain multiple
          // times.
          GrGeomOutLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
          {
            ipx = SubvectorEltIndex(perm_x_sub, i, j, k);
            ipy = SubvectorEltIndex(perm_y_sub, i, j, k);
            ipz = SubvectorEltIndex(perm_z_sub, i, j, k);

            perm_x_dat[ipx] = 0.0;
            perm_y_dat[ipy] = 0.0;
            perm_z_dat[ipz] = 0.0;
          });
        }    /* End subgrid loop */
      }      /* End loop over regions */

      break;
    }        /* End case 0 */

    case 1:  /* kx, ky, kz specified cell-wise by pfb files */
      /* sk: I replaced the function/macro PFVPROD(), which didn't work, with the common loop over
       *     the subgrids and geometries 01/20/2005*/
    {
      Vector  *kx_values, *ky_values, *kz_values;

      dummy1 = (Type1*)(public_xtra->data);

      kx_values = dummy1->kx_values;
      ky_values = dummy1->ky_values;
      kz_values = dummy1->kz_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(sg, GridSubgrids(grid))
      {
        subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

        perm_x_sub = VectorSubvector(perm_x, sg);
        perm_y_sub = VectorSubvector(perm_y, sg);
        perm_z_sub = VectorSubvector(perm_z, sg);

        kx_values_sub = VectorSubvector(kx_values, sg);
        ky_values_sub = VectorSubvector(ky_values, sg);
        kz_values_sub = VectorSubvector(kz_values, sg);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        r = SubgridRX(subgrid);

        perm_x_dat = SubvectorData(perm_x_sub);
        perm_y_dat = SubvectorData(perm_y_sub);
        perm_z_dat = SubvectorData(perm_z_sub);

        kx_values_dat = SubvectorData(kx_values_sub);
        ky_values_dat = SubvectorData(ky_values_sub);
        kz_values_dat = SubvectorData(kz_values_sub);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ipx = SubvectorEltIndex(kx_values_sub, i, j, k);
          ipy = SubvectorEltIndex(ky_values_sub, i, j, k);
          ipz = SubvectorEltIndex(ky_values_sub, i, j, k);

          perm_x_dat[ipx] = perm_z_dat[ipz] * kx_values_dat[ipx];
          perm_y_dat[ipy] = perm_z_dat[ipz] * ky_values_dat[ipy];
          perm_z_dat[ipz] = perm_z_dat[ipz] * kz_values_dat[ipz];
        });
      }      /* End subgrid loop */
      break;
    }        /* End case 1 */
  }          /* End switch statement */


  /*-----------------------------------------------------------------------
   * exchange boundary data for cell permeability values
   *-----------------------------------------------------------------------*/
  handle = InitVectorUpdate(perm_x, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(perm_y, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(perm_z, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /*------------------------------------------------------------------------
   * Compute an average permeability for each flux well
   *------------------------------------------------------------------------*/

  if (WellDataNumFluxWells(well_data) > 0)
  {
    grid = VectorGrid(perm_x);

    subgrids = GridSubgrids(grid);

    for (well = 0; well < WellDataNumFluxWells(well_data); well++)
    {
      well_data_physical = WellDataFluxWellPhysical(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      well_volume = WellDataPhysicalSize(well_data_physical);

      perm_average_x = 0.0;
      perm_average_y = 0.0;
      perm_average_z = 0.0;

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        perm_x_sub = VectorSubvector(perm_x, sg);
        perm_y_sub = VectorSubvector(perm_y, sg);
        perm_z_sub = VectorSubvector(perm_z, sg);

        nx_p = SubvectorNX(perm_x_sub);
        ny_p = SubvectorNY(perm_x_sub);
        nz_p = SubvectorNZ(perm_x_sub);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          dx = SubgridDX(subgrid);
          dy = SubgridDY(subgrid);
          dz = SubgridDZ(subgrid);

          cell_volume = dx * dy * dz;

          perm_x_elt = SubvectorElt(perm_x_sub, ix, iy, iz);
          perm_y_elt = SubvectorElt(perm_y_sub, ix, iy, iz);
          perm_z_elt = SubvectorElt(perm_z_sub, ix, iy, iz);

          pi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    pi, nx_p, ny_p, nz_p, 1, 1, 1,
          {
            perm_average_x += perm_x_elt[pi] * (cell_volume / well_volume);
            perm_average_y += perm_y_elt[pi] * (cell_volume / well_volume);
            perm_average_z += perm_z_elt[pi] * (cell_volume / well_volume);
          });

          FreeSubgrid(tmp_subgrid);       /* done with temporary subgrid */
        }
      }

      result_invoice = amps_NewInvoice("%d", &perm_average_x);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      result_invoice = amps_NewInvoice("%d", &perm_average_y);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      result_invoice = amps_NewInvoice("%d", &perm_average_z);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      if (perm_average_x > 0.0)
      {
        WellDataPhysicalAveragePermeabilityX(well_data_physical)
          = perm_average_x;
      }
      else
      {
        WellDataPhysicalAveragePermeabilityX(well_data_physical) = 1.0;
      }

      if (perm_average_y > 0.0)
      {
        WellDataPhysicalAveragePermeabilityY(well_data_physical)
          = perm_average_y;
      }
      else
      {
        WellDataPhysicalAveragePermeabilityY(well_data_physical) = 1.0;
      }

      if (perm_average_z > 0.0)
      {
        WellDataPhysicalAveragePermeabilityZ(well_data_physical)
          = perm_average_z;
      }
      else
      {
        WellDataPhysicalAveragePermeabilityZ(well_data_physical) = 1.0;
      }
    }
  }

  EndTiming(public_xtra->time_index);

  tfree(cdata);

  return;
}


/*--------------------------------------------------------------------------
 * SubsrfSimInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SubsrfSimInitInstanceXtra(
                                     Grid *  grid,
                                     double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  Type1     *dummy1;

  int num_geo_indexes = (public_xtra->num_geo_indexes);
  int i;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    (instance_xtra->grid) = grid;

    /* Use a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);

      dummy1->kx_values = NewVectorType(grid, 1, 1, vector_cell_centered);

      ReadPFBinary((dummy1->kx_file_name),
                   (dummy1->kx_values));
    }
  }

  if (grid != NULL)
  {
    (instance_xtra->grid) = grid;

    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);

      dummy1->ky_values = NewVectorType(grid, 1, 1, vector_cell_centered);

      ReadPFBinary((dummy1->ky_file_name),
                   (dummy1->ky_values));
    }
  }

  if (grid != NULL)
  {
    (instance_xtra->grid) = grid;

    /* Use a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);

      dummy1->kz_values = NewVectorType(grid, 1, 1, vector_cell_centered);

      ReadPFBinary((dummy1->kz_file_name),
                   (dummy1->kz_values));
    }
  }


  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->KFieldSimulators) = talloc(PFModule*, num_geo_indexes);

    for (i = 0; i < num_geo_indexes; i++)
      (instance_xtra->KFieldSimulators)[i] =
        PFModuleNewInstanceType(KFieldSimulatorInitInstanceXtraInvoke,
                                (public_xtra->KFieldSimulators)[i],
                                (grid, temp_data));
  }
  else
  {
    for (i = 0; i < num_geo_indexes; i++)
      PFModuleReNewInstanceType(KFieldSimulatorInitInstanceXtraInvoke,
                                (instance_xtra->KFieldSimulators)[i],
                                (grid, temp_data));
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SubsrfSimFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SubsrfSimFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Type1 *dummy1;
  int i;

  if (public_xtra->type == 1)
  {
    dummy1 = (Type1*)(public_xtra->data);
    FreeVector(dummy1->kx_values);
    FreeVector(dummy1->ky_values);
    FreeVector(dummy1->kz_values);
  }


  if (instance_xtra)
  {
    for (i = 0; i < (public_xtra->num_geo_indexes); i++)
      PFModuleFreeInstance(instance_xtra->KFieldSimulators[i]);
    tfree(instance_xtra->KFieldSimulators);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SubsrfSimNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SubsrfSimNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;

  amps_Invoice invoice;
  amps_File amps_cdata_file;

  char          *filename;
  int n, nc;
  double x, y, z, v;
  int i, sim_type;
  int num_geo_indexes;

  char          *geo_index_names;
  NameArray geo_index_na;

  char          *sim_type_name;
  char          *switch_name;
  char key[IDB_MAX_KEY_LEN];

  NameArray switch_na, type_na;

  char *geom_name;

  public_xtra = ctalloc(PublicXtra, 1);

  /*--------------------------------------------------------------
   * Is there conditioning data? If so, get the file name and
   * read the data into appropriate arrays.
   *--------------------------------------------------------------*/

  filename = GetStringDefault("Perm.Conditioning.FileName", "NA");

  /* If user has specified a filename do conditioning */
  if (strcmp(filename, "NA"))
  {
    /* Open the new file containing the permeability data */
    if ((amps_cdata_file = amps_SFopen(filename, "r")) == NULL)
    {
      amps_Printf("Error: can't open file %s\n", filename);
      exit(1);
    }

    /* Get the number of data points and allocate memory */
    invoice = amps_NewInvoice("%i", &nc);
    amps_SFBCast(amps_CommWorld, amps_cdata_file, invoice);
    amps_FreeInvoice(invoice);
    (public_xtra->nc) = nc;

    /* Allocate memory for the data arrays */
    (public_xtra->x) = ctalloc(double, nc);
    (public_xtra->y) = ctalloc(double, nc);
    (public_xtra->z) = ctalloc(double, nc);
    (public_xtra->v) = ctalloc(double, nc);

    invoice = amps_NewInvoice("%d%d%d%d", &x, &y, &z, &v);
    for (n = 0; n < nc; n++)
    {
      amps_SFBCast(amps_CommWorld, amps_cdata_file, invoice);
      (public_xtra->x)[n] = x;
      (public_xtra->y)[n] = y;
      (public_xtra->z)[n] = z;
      (public_xtra->v)[n] = v;
    }
    amps_FreeInvoice(invoice);
    amps_SFclose(amps_cdata_file);
  }

  else
  {
    /* If there's no data file, the number of conditioning points is zero. */
    (public_xtra->nc) = 0;
  }


  /*--------------------------------------------------------------
   * Read in the number of geounits to simulate (num_geo_indexes),
   * and assign properties to each
   *--------------------------------------------------------------*/

  geo_index_names = GetString("Geom.Perm.Names");
  geo_index_na = (public_xtra->geo_indexes_na) =
    NA_NewNameArray(geo_index_names);
  num_geo_indexes = NA_Sizeof(geo_index_na);

  (public_xtra->num_geo_indexes) = num_geo_indexes;
  (public_xtra->geo_indexes) = ctalloc(int, num_geo_indexes);
  (public_xtra->KFieldSimulators) = ctalloc(PFModule*, num_geo_indexes);

  switch_na = NA_NewNameArray("Constant TurnBands ParGauss PFBFile");

  for (i = 0; i < num_geo_indexes; i++)
  {
    /* Input geounit index (ind) and simulation method type (sim_type) */
    geom_name = NA_IndexToName(geo_index_na, i);

    public_xtra->geo_indexes[i] = NA_NameToIndex(GlobalsGeomNames,
                                                 geom_name);

    if (public_xtra->geo_indexes[i] < 0)
    {
      InputError("Error: invalid geometry name <%s> for key <%s>\n",
                 geom_name, "Geom.Perm.Names");
    }

    sprintf(key, "Geom.%s.Perm.Type", geom_name);
    sim_type_name = GetString(key);

    sim_type = NA_NameToIndexExitOnError(switch_na, sim_type_name, key);

    /* Assign the K field simulator method and invoke the "New" function */
    switch (sim_type)
    {
      case 0:
      {
        (public_xtra->KFieldSimulators)[i] =
          PFModuleNewModuleType(KFieldSimulatorNewPublicXtra, ConstantRF, (geom_name));
        break;
      }

      case 1:
      {
        (public_xtra->KFieldSimulators)[i] =
          PFModuleNewModuleType(KFieldSimulatorNewPublicXtra, TurningBandsRF, (geom_name));
        break;
      }

      case 2:
      {
        (public_xtra->KFieldSimulators)[i] =
          PFModuleNewModuleType(KFieldSimulatorNewPublicXtra, PGSRF, (geom_name));
        break;
      }

      case 3:
      {
        (public_xtra->KFieldSimulators)[i] =
          PFModuleNewModuleType(KFieldSimulatorNewPublicXtra, InputRF, (geom_name));
        break;
      }

      default:
      {
        InputError("Invalid switch value <%s> for key <%s>", sim_type_name, key);
      }
    }
  }

  type_na = NA_NewNameArray("TensorByGeom TensorByFile");
  switch_name = GetString("Perm.TensorType");
  public_xtra->type = NA_NameToIndexExitOnError(type_na, switch_name, "TensorByGeom TensorByFile");

  switch (public_xtra->type)
  {
    case 0:
    {
      int num_tens_indexes;

      dummy0 = ctalloc(Type0, 1);

      geo_index_names = GetString("Geom.Perm.TensorByGeom.Names");
      geo_index_na = dummy0->tens_indexes_na
                       = NA_NewNameArray(geo_index_names);
      num_tens_indexes = NA_Sizeof(geo_index_na);

      dummy0->num_tens_indexes = num_tens_indexes;
      dummy0->tens_indexes = ctalloc(int, num_tens_indexes);

      dummy0->kx_values = ctalloc(double, num_tens_indexes);
      dummy0->ky_values = ctalloc(double, num_tens_indexes);
      dummy0->kz_values = ctalloc(double, num_tens_indexes);

      for (i = 0; i < num_tens_indexes; i++)
      {
        geom_name = NA_IndexToName(geo_index_na, i);

        dummy0->tens_indexes[i] = NA_NameToIndex(GlobalsGeomNames,
                                                 geom_name);

        sprintf(key, "Geom.%s.Perm.TensorValX", geom_name);
        dummy0->kx_values[i] = GetDouble(key);

        sprintf(key, "Geom.%s.Perm.TensorValY", geom_name);
        dummy0->ky_values[i] = GetDouble(key);

        sprintf(key, "Geom.%s.Perm.TensorValZ", geom_name);
        dummy0->kz_values[i] = GetDouble(key);
      }

      public_xtra->data = (void*)dummy0;

      break;
    }

    case 1:
    {
      dummy1 = ctalloc(Type1, 1);

      sprintf(key, "Geom.%s.Perm.TensorFileX", "domain");
      dummy1->kx_file_name = GetString(key);

      sprintf(key, "Geom.%s.Perm.TensorFileY", "domain");
      dummy1->ky_file_name = GetString(key);

      sprintf(key, "Geom.%s.Perm.TensorFileZ", "domain");
      dummy1->kz_file_name = GetString(key);

      public_xtra->data = (void*)dummy1;

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }   /*End switch */

  (public_xtra->time_index) = RegisterTiming("SubsrfSim");

  NA_FreeNameArray(type_na);
  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SubsrfSimFreePublicXtra
 *--------------------------------------------------------------------------*/

void  SubsrfSimFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;
  Type1       *dummy1;

  int i;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->geo_indexes_na);

    for (i = 0; i < (public_xtra->num_geo_indexes); i++)
      PFModuleFreeModule(public_xtra->KFieldSimulators[i]);
    tfree(public_xtra->KFieldSimulators);

    if ((public_xtra->nc) > 0)
    {
      tfree(public_xtra->x);
      tfree(public_xtra->y);
      tfree(public_xtra->z);
      tfree(public_xtra->v);
    }

    if (public_xtra->type == 0)
    {
      dummy0 = (Type0*)public_xtra->data;

      NA_FreeNameArray(dummy0->tens_indexes_na);
      tfree(dummy0->tens_indexes);

      tfree(dummy0->kx_values);
      tfree(dummy0->ky_values);
      tfree(dummy0->kz_values);

      tfree(dummy0);
    }

    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)public_xtra->data;
      tfree(dummy1);
    }

    tfree(public_xtra->geo_indexes);
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SubsrfSimSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SubsrfSimSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;

  int n;

  /* set `sz' to max of each of the called modules */
  for (n = 0; n < (public_xtra->num_geo_indexes); n++)
    sz = pfmax(sz,
               PFModuleSizeOfTempData((instance_xtra->KFieldSimulators)[n]));

  return sz;
}
