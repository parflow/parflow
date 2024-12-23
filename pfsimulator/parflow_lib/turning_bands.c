/*****************************************************************************
* Turn, InitTurn, FreeTurn, NewTurn
*
* Routines to generate a Gaussian random field.
*
*
* (C) 1993 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Log: turning_bands.c,v $
* Revision 1.1.1.1  2006/02/14 23:05:51  kollet
* CLM.PF_1.0
*
* Revision 1.1.1.1  2006/02/14 18:51:22  kollet
* CLM.PF_1.0
*
* Revision 1.9  1997/09/09 20:06:13  ssmith
* Added additional input checking
* Fixed a few problems with the new input file format
*
* Revision 1.8  1997/09/03 17:43:48  ssmith
* New input file format
*
* Revision 1.7  1994/07/15 20:33:34  ssmith
* Added timing routines
*
* Revision 1.6  1994/07/01  20:28:09  falgout
* Added multigrid and made major revisions
*
* Revision 1.5  1994/03/03  01:47:32  ssmith
* Fixed mallocs, use modules instead of methods
*
* Revision 1.4  1994/02/10  00:07:35  ssmith
* Updated to use AMPS
*
* Revision 1.3  1994/01/06  01:20:07  falgout
* Modified to work better for anisotropic problems.
* The definition of dzeta is now given in terms of an input rzeta value.
*
* Revision 1.2  1993/12/22  17:22:05  falgout
* The initial seed value is now read in from the input_file (turning bands).
*
* Revision 1.1  1993/07/23  20:42:11  falgout
* Initial revision
*
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"
#include <cmath>

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

  double cmin;          /* NEED? */
  double cmax;          /* NEED? */
  int ndelt;            /* NEED? */
  int ks1;              /* NEED? */
  int ks2;              /* NEED? */
  int ks3;              /* NEED? */

  int log_normal;

  int seed;

  int time_index;
} TurnXtra;


/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

#define Index(z, zm, dz, i) ((int)((z - zm) / dz + 0.5) - i)

/*--------------------------------------------------------------------------
 * Turn
 *--------------------------------------------------------------------------*/

void Turn(field, vxtra)
Vector * field;
void   *vxtra;
{
  TurnXtra *xtra = vxtra;

  double lambdaX = (xtra->lambdaX);
  double lambdaY = (xtra->lambdaY);
  double lambdaZ = (xtra->lambdaZ);
  double mean = (xtra->mean);
  double sigma = (xtra->sigma);
  int num_lines = (xtra->num_lines);
  double rzeta = (xtra->rzeta);
  double Kmax = (xtra->Kmax);
  double dK = (xtra->dK);
  int log_normal = (xtra->log_normal);

  double pi = acos(-1.0);

  Grid     *grid = VectorGrid(field);

  double NX = GridNX(grid) + 2;
  double NY = GridNY(grid) + 2;
  double NZ = GridNZ(grid) + 2;

  double DX = GridDX(grid);
  double DY = GridDY(grid);
  double DZ = GridDZ(grid);

  Subgrid   *subgrid;

  Subvector *sub_field;

  double xt, yt, zt;      /* translated x, y, z */
  int nx, ny, nz;
  double dx, dy, dz;

  double phi, theta;
  double   *theta_array, *phi_array;

  double unitx, unity, unitz;

  double zeta, dzeta;
  double ZetaMin;
  int NZeta;
  int izeta, nzeta;

  double   *Z;

  int l;
  int gridloop;
  int i, j, k;
  double x, y, z;
  double   *field_ptr;
  double sqrtnl;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/
  BeginTiming(xtra->time_index);

  /*-----------------------------------------------------------------------
   * start turning bands algorithm
   *-----------------------------------------------------------------------*/

  /* initialize random number generator */
  SeedRand(xtra->seed);

  /* transform 3-space so that correlation lengths are normalized to 1 */
  DX = DX / lambdaX;
  DY = DY / lambdaY;
  DZ = DZ / lambdaZ;
  dzeta = pfmin(pfmin(DX, DY), DZ) / rzeta;   /* THIS WILL BE MODIFIED SOON */

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
   * START grid loop
   *-----------------------------------------------------------------------*/

  for (gridloop = 0; gridloop < GridNumSubgrids(grid); gridloop++)
  {
    subgrid = GridSubgrid(grid, gridloop);

    sub_field = VectorSubvector(field, gridloop);

    nx = SubgridNX(subgrid) + 2;
    ny = SubgridNY(subgrid) + 2;
    nz = SubgridNZ(subgrid) + 2;

    dx = DX / pow(2.0, (double)SubgridRX(subgrid));
    dy = DY / pow(2.0, (double)SubgridRY(subgrid));
    dz = DZ / pow(2.0, (double)SubgridRZ(subgrid));

    xt = DX * (SubgridIX(subgrid) + 1) - dx;  /* xt = (x-dx) - (X-DX) */
    yt = DY * (SubgridIY(subgrid) + 1) - dy;  /* yt = (y-dy) - (Y-DY) */
    zt = DZ * (SubgridIZ(subgrid) + 1) - dz;  /* zt = (z-dz) - (Z-DZ) */

    /* malloc space for Z */
    nzeta = (sqrt(pow(((double)nx) * DX, 2.0) +
                  pow(((double)ny) * DY, 2.0) +
                  pow(((double)nz) * DZ, 2.0)) / dzeta) + 2;
    Z = talloc(double, nzeta);

    /* zero the field subvector */
    field_ptr = SubvectorCoeff(sub_field, 0, -1, -1, -1);
    for (i = 0; i < (nx * ny * nz); i++)
      *(field_ptr++) = 0.0;

    /*--------------------------------------------------------------------
     * START line loop
     *--------------------------------------------------------------------*/

    for (l = 0; l < num_lines; l++)
    {
      /* determine phi, theta (line direction) */
      theta = theta_array[l];
      phi = phi_array[l];

      /* compute unitx, unity, unitz */
      unitx = cos(theta) * sin(phi);
      unity = sin(theta) * sin(phi);
      unitz = cos(phi);

      /* determine ZetaMin, Nzeta, izeta, and nzeta */
      ZetaMin = (pfmin(0, NX * DX * unitx) +
                 pfmin(0, NY * DY * unity) +
                 pfmin(0, NZ * DZ * unitz));
      NZeta = (fabs(NX * DX * unitx) +
               fabs(NY * DY * unity) +
               fabs(NZ * DZ * unitz)) / dzeta + 2;

      zeta = (pfmin(xt * unitx, (xt + nx * dx) * unitx) +
              pfmin(yt * unity, (yt + ny * dy) * unity) +
              pfmin(zt * unitz, (zt + nz * dz) * unitz));
      izeta = Index(zeta, ZetaMin, dzeta, 0);
      nzeta = (fabs(nx * dx * unitx) +
               fabs(ny * dy * unity) +
               fabs(nz * dz * unitz)) / dzeta + 2;

      /*--------------------------------------------*/
      /* get the line process, Z */

      LineProc(Z, phi, theta, ZetaMin, NZeta, dzeta, izeta, nzeta,
               Kmax, dK);

      /*--------------------------------------------*/
      /* project Z onto field */

      field_ptr = SubvectorCoeff(sub_field, 0, -1, -1, -1);

      z = zt;
      for (k = 0; k < nz; k++)
      {
        y = yt;
        for (j = 0; j < ny; j++)
        {
          x = xt;
          for (i = 0; i < nx; i++)
          {
            zeta = x * unitx + y * unity + z * unitz;
            *(field_ptr++) += Z[Index(zeta, ZetaMin, dzeta, izeta)];

            x += dx;
          }
          y += dy;
        }
        z += dz;
      }
    }

    /*--------------------------------------------------------------------
     * END line loop
     *--------------------------------------------------------------------*/

    /* scale field by sqrt(num_lines) */
    sqrtnl = 1.0 / sqrt((double)num_lines);
    field_ptr = SubvectorCoeff(sub_field, 0, -1, -1, -1);
    for (i = 0; i < (nx * ny * nz); i++)
      *(field_ptr++) *= sqrtnl;

    /* if (log_normal == 1) transform field */
    if (log_normal == 1)
    {
      field_ptr = SubvectorCoeff(sub_field, 0, -1, -1, -1);
      for (i = 0; i < (nx * ny * nz); i++)
      {
        *field_ptr = mean * exp((sigma) * (*field_ptr));
        field_ptr++;
      }
    }

    /* free up the space for Z */
    free(Z);
  }

  free(theta_array);
  free(phi_array);

  /*-----------------------------------------------------------------------
   * END grid loop
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  EndTiming(xtra->time_index);
}


/*--------------------------------------------------------------------------
 * InitTurn *** DON'T NEED ***
 *--------------------------------------------------------------------------*/

int InitTurn()
{
  return 0;
}


/*--------------------------------------------------------------------------
 * NewTurn
 *--------------------------------------------------------------------------*/

void *NewTurn(geom_name)
char *geom_name;
{
  TurnXtra *xtra;

  xtra = ctalloc(TurnXtra, 1);

  /*-------------------------------------------------------------*/
  /* receive and setup user input parameters */

  sprintf(key, "Geom.%s.Perm.LambdaX", geom_name);
  xtra->lambdaX = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LambdaY", geom_name);
  xtra->lambdaY = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LambdaZ", geom_name);
  xtra->lambdaZ = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.GeomMean", geom_name);
  xtra->mean = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.sigma", geom_name);
  xtra->sigma = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.NumLines", geom_name);
  xtra->NumLines = GetInt(key);

  sprintf(key, "Geom.%s.Perm.RZeta", geom_name);
  xtra->rzeta = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.KMax", geom_name);
  xtra->Kmax = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.DelK", geom_name);
  xtra->dK = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LogNormal", geom_name);
  tmp = GetString(key);

  if (strcmp(tmp, "True"))
    xtra->log_normal = 1;
  else if (strcmp(tmp, "False"))
    xtra->log_normal = 0;
  else
  {
    NA_InputError("Error: Invalid True/False value for key <%s> was <%s>\n",
                  key, tmp);
  }

  sprintf(key, "Geom.%s.Perm.Seed", geom_name);
  xtra->seed = GetInt(key);

  (xtra->time_index) = RegisterTiming("Turning Bands");

  /*-------------------------------------------------------------*/

  return (void*)xtra;
}


/*--------------------------------------------------------------------------
 * FreeTurn
 *--------------------------------------------------------------------------*/

void FreeTurn(xtra)
void *xtra;
{
  free(xtra);
}



