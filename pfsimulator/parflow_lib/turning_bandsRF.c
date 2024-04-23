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
* Routines to generate a Gaussian random field.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  double lambdaX;
  double lambdaY;
  double lambdaZ;
  double mean;
  double sigma;
  int num_lines;
  double rzeta;
  double Kmax;
  double dK;
  int log_normal;
  double low_cutoff;
  double high_cutoff;
  int seed;
  int strat_type;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid   *grid;
  double *temp_data;

  /* Instance data */
} InstanceXtra;


/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

#define Index(z, dz) ((int)(z / dz + 0.5))

/*--------------------------------------------------------------------------
 * TurningBandsRF:
 *   Note: The indices on each line are numbered so that index 0 lines
 *   up with the (X,Y,Z) background coordinate, which is treated as
 *   the origin in this routine.  So, if (X,Y,Z) is NOT (0,0,0, this
 *   routine will not generate the same field that Andy Tompson's turn3d
 *   code will generate.
 *--------------------------------------------------------------------------*/

void          TurningBandsRF(
                             GeomSolid *  geounit,
                             GrGeomSolid *gr_geounit,
                             Vector *     field,
                             RFCondData * cdata)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  double lambdaX = (public_xtra->lambdaX);
  double lambdaY = (public_xtra->lambdaY);
  double lambdaZ = (public_xtra->lambdaZ);
  double mean = (public_xtra->mean);
  double sigma = (public_xtra->sigma);
  int num_lines = (public_xtra->num_lines);
  double rzeta = (public_xtra->rzeta);
  double Kmax = (public_xtra->Kmax);
  double dK = (public_xtra->dK);
  int log_normal = (public_xtra->log_normal);
  int strat_type = (public_xtra->strat_type);
  double low_cutoff = (public_xtra->low_cutoff);
  double high_cutoff = (public_xtra->high_cutoff);

  double pi = acos(-1.0);

  Grid       *grid = (instance_xtra->grid);

  Subgrid    *subgrid;
  Subvector  *field_sub;

  double xlo, ylo, zlo, sh_zlo;
  double xhi, yhi, zhi, sh_zhi;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  double dx, dy, dz;

  double phi, theta;
  double     *theta_array, *phi_array;

  double unitx, unity, unitz;

  double    **shear_arrays, *shear_array;
  double     *shear_min, *shear_max;

  double zeta, dzeta;
  int izeta, nzeta;

  double     *Z;

  int is, l, i, j, k;
  int index;
  int doing_TB;
  double x, y, z;
  double     *fieldp;
  double sqrtnl;

  Statistics *stats;


  /*-----------------------------------------------------------------------
   * start turning bands algorithm
   *-----------------------------------------------------------------------*/

  /* initialize random number generator */
  SeedRand(public_xtra->seed);

  /* malloc space for theta_array and phi_array */
  theta_array = talloc(double, num_lines);
  phi_array = talloc(double, num_lines);

  /* compute line directions */
  for (l = 0; l < num_lines; l++)
  {
    theta_array[l] = 2.0 * pi * Rand();
    phi_array[l] = acos(1.0 - 2.0 * Rand());
  }

  /*-----------------------------------------------------------------------
   * Determine by how much to shear the field:
   *   If there is no GeomSolid representation of the geounit, then
   *   we do regular turning bands (by setting the shear_arrays to
   *   all zeros).
   *-----------------------------------------------------------------------*/

  /* Do regular turning bands */
  if ((strat_type == 0) || (!geounit))
  {
    shear_arrays = ctalloc(double *, GridNumSubgrids(grid));
    shear_min = ctalloc(double, GridNumSubgrids(grid));
    shear_max = ctalloc(double, GridNumSubgrids(grid));

    ForSubgridI(is, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, is);

      shear_arrays[is] =
        ctalloc(double, (SubgridNX(subgrid) * SubgridNY(subgrid)));
    }
  }

  /* Let the stratification follow the geounits */
  else
  {
    shear_arrays = SimShear(&shear_min, &shear_max,
                            geounit, GridSubgrids(grid), strat_type);
  }

  /*-----------------------------------------------------------------------
   * START grid loop
   *-----------------------------------------------------------------------*/

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);
    field_sub = VectorSubvector(field, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolutions are the same in all 3 directions */
    r = SubgridRX(subgrid);

    /* Note: lines run through the grid anchor point */
    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    xlo = SubgridX(subgrid);
    ylo = SubgridY(subgrid);
    zlo = SubgridZ(subgrid);
    xhi = xlo + (nx - 1) * dx;
    yhi = ylo + (ny - 1) * dy;
    zhi = zlo + (nz - 1) * dz;

    /* Scale by correlation lengths */
    dx /= lambdaX;
    dy /= lambdaY;
    dz /= lambdaZ;
    xlo /= lambdaX;
    ylo /= lambdaY;
    zlo /= lambdaZ;
    xhi /= lambdaX;
    yhi /= lambdaY;
    zhi /= lambdaZ;

    dzeta = pfmin(pfmin(dx, dy), dz) / rzeta;    /* THIS WILL BE MODIFIED SOON */

    /*--------------------------------------------------------------------
     * Check to see if we need to do TB and zero out the field.
     *--------------------------------------------------------------------*/

    fieldp = SubvectorData(field_sub);
    doing_TB = 0;
    GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
    {
      index = SubvectorEltIndex(field_sub, i, j, k);

      fieldp[index] = 0.0;
      doing_TB = 1;
    });

    if (!doing_TB)
      continue;

    /*--------------------------------------------------------------------
     * Scale the shear_array to the TB coordinate system.
     *
     * Modify the subgrid z extent (zhi) to account for the shearing
     * to be done.  This is necessary to get subsequent line size
     * computations correct.
     *--------------------------------------------------------------------*/

    shear_array = shear_arrays[is];

    /* Scale shear elevations */
    for (i = 0; i < (nx * ny); i++)
      shear_array[i] /= lambdaZ;
    shear_min[is] /= lambdaZ;
    shear_max[is] /= lambdaZ;

    /* Compute "shear" zlo and zhi values */
    sh_zlo = zlo - shear_max[is];
    sh_zhi = zhi - shear_min[is];

    /*--------------------------------------------------------------------
     * Generate lines
     *--------------------------------------------------------------------*/

    /* malloc space for Z */
    nzeta = (int)((sqrt(pow((xhi - xlo), 2.0) +
                        pow((yhi - ylo), 2.0) +
                        pow((sh_zhi - sh_zlo), 2.0)) / dzeta)) + 2;
    Z = talloc(double, nzeta);

    for (l = 0; l < num_lines; l++)
    {
      /* determine phi, theta (line direction) */
      theta = theta_array[l];
      phi = phi_array[l];

      /* compute unitx, unity, unitz */
      unitx = cos(theta) * sin(phi);
      unity = sin(theta) * sin(phi);
      unitz = cos(phi);

      /* determine izeta, and nzeta */
      zeta = (pfmin(xlo * unitx, xhi * unitx) +
              pfmin(ylo * unity, yhi * unity) +
              pfmin(sh_zlo * unitz, sh_zhi * unitz));
      izeta = Index(zeta, dzeta);
      nzeta = (int)((fabs((xhi - xlo) * unitx) +
                     fabs((yhi - ylo) * unity) +
                     fabs((sh_zhi - sh_zlo) * unitz)) / dzeta) + 2;

      /* Get the line process, Z */
      LineProc(Z, phi, theta, dzeta, izeta, nzeta, Kmax, dK);

      /* Project Z onto field */
      fieldp = SubvectorData(field_sub);
      GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
      {
        index = SubvectorEltIndex(field_sub, i, j, k);

        x = xlo + (i - ix) * dx;
        y = ylo + (j - iy) * dy;
        z = zlo + (k - iz) * dz - shear_array[(j - iy) * nx + (i - ix)];
        zeta = x * unitx + y * unity + z * unitz;
        fieldp[index] += Z[Index(zeta, dzeta) - izeta];
      });
    }

    /*--------------------------------------------------------------------
     * Scale by square root of number of lines, condition, and shift
     *--------------------------------------------------------------------*/

    /* scale field by sqrt(num_lines) */
    sqrtnl = 1.0 / sqrt((double)num_lines);
    fieldp = SubvectorData(field_sub);
    GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
    {
      index = SubvectorEltIndex(field_sub, i, j, k);
      fieldp[index] *= sqrtnl;
    });

    /*
     * Condition the field to data using the p-field method.
     * See "SCRF 1993 Annual Report", Stanford Center for
     * Reservoir Forecasting.
     */
    if (cdata->nc)
    {
      stats = ctalloc(Statistics, 1);
      (stats->mean) = mean;
      (stats->sigma) = sigma;
      (stats->lambdaX) = lambdaX;
      (stats->lambdaY) = lambdaY;
      (stats->lambdaZ) = lambdaZ;
      (stats->lognormal) = log_normal;
      PField(grid, geounit, gr_geounit, field, cdata, stats);
    }

    /* make field normal or lognormal */
    switch (log_normal)
    {
      case 0:    /* normal distribution */
        GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
      {
        index = SubvectorEltIndex(field_sub, i, j, k);
        fieldp[index] = mean + sigma * fieldp[index];
      });
        break;

      case 1:    /* log normal distribution */
        GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
      {
        index = SubvectorEltIndex(field_sub, i, j, k);
        fieldp[index] = mean * exp((sigma) * fieldp[index]);
      });
        break;

      case 2:    /* normal distribution with low and high cutoffs */
        GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
      {
        index = SubvectorEltIndex(field_sub, i, j, k);
        fieldp[index] = mean + sigma * fieldp[index];
        if (fieldp[index] < low_cutoff)
          fieldp[index] = low_cutoff;
        if (fieldp[index] > high_cutoff)
          fieldp[index] = high_cutoff;
      });
        break;

      case 3:    /* log normal distribution with low and high cutoffs */
        GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
      {
        index = SubvectorEltIndex(field_sub, i, j, k);
        fieldp[index] = mean * exp((sigma) * fieldp[index]);
        if (fieldp[index] < low_cutoff)
          fieldp[index] = low_cutoff;
        if (fieldp[index] > high_cutoff)
          fieldp[index] = high_cutoff;
      });
        break;
    }   /* end switch(lognormal)  */

    /* clean up */
    tfree(shear_array);
    tfree(Z);
  }

  tfree(shear_arrays);
  tfree(shear_min);
  tfree(shear_max);
  tfree(theta_array);
  tfree(phi_array);

  /*-----------------------------------------------------------------------
   * END grid loop
   *-----------------------------------------------------------------------*/
}


/*--------------------------------------------------------------------------
 * TurningBandsRFInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *TurningBandsRFInitInstanceXtra(
                                          Grid *  grid,
                                          double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */

    /* set new data */
    (instance_xtra->grid) = grid;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * TurningBandsRFFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  TurningBandsRFFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * TurningBandsRFNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *TurningBandsRFNewPublicXtra(char *geom_name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  NameArray log_normal_na;
  NameArray strat_type_na;

  char *tmp;

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "Geom.%s.Perm.LambdaX", geom_name);
  public_xtra->lambdaX = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LambdaY", geom_name);
  public_xtra->lambdaY = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LambdaZ", geom_name);
  public_xtra->lambdaZ = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.GeomMean", geom_name);
  public_xtra->mean = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.Sigma", geom_name);
  public_xtra->sigma = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.NumLines", geom_name);
  public_xtra->num_lines = GetIntDefault(key, 100);

  sprintf(key, "Geom.%s.Perm.RZeta", geom_name);
  public_xtra->rzeta = GetDoubleDefault(key, 5.0);
  sprintf(key, "Geom.%s.Perm.KMax", geom_name);
  public_xtra->Kmax = GetDoubleDefault(key, 100.0);

  sprintf(key, "Geom.%s.Perm.DelK", geom_name);
  public_xtra->dK = GetDoubleDefault(key, 0.2);

  log_normal_na = NA_NewNameArray("Normal Log NormalTruncated LogTruncated");
  sprintf(key, "Geom.%s.Perm.LogNormal", geom_name);
  tmp = GetStringDefault(key, "LogTruncated");
  public_xtra->log_normal = NA_NameToIndexExitOnError(log_normal_na, tmp, key);
  NA_FreeNameArray(log_normal_na);

  sprintf(key, "Geom.%s.Perm.Seed", geom_name);
  public_xtra->seed = GetIntDefault(key, 1);

  strat_type_na = NA_NewNameArray("Horizontal Bottom Top");
  sprintf(key, "Geom.%s.Perm.StratType", geom_name);
  tmp = GetStringDefault(key, "Bottom");
  public_xtra->strat_type = NA_NameToIndexExitOnError(strat_type_na, tmp, key);
  NA_FreeNameArray(strat_type_na);

  if (public_xtra->log_normal > 1)
  {
    sprintf(key, "Geom.%s.Perm.LowCutoff", geom_name);
    public_xtra->low_cutoff = GetDouble(key);

    sprintf(key, "Geom.%s.Perm.HighCutoff", geom_name);
    public_xtra->high_cutoff = GetDouble(key);
  }

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * TurningBandsRFFreePublicXtra
 *--------------------------------------------------------------------------*/

void  TurningBandsRFFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * TurningBandsRFSizeOfTempData
 *--------------------------------------------------------------------------*/

int  TurningBandsRFSizeOfTempData()
{
  return 0;
}
