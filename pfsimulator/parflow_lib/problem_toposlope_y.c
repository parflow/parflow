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

#include "parflow.h"
#include "parflow_netcdf.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int type;
  void  *data;
} PublicXtra;

typedef struct {
  Grid   *grid3d;
  Grid   *grid2d;
} InstanceXtra;

typedef struct {
  NameArray regions;
  int num_regions;


  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
  int function_type;
} Type1;                       /* Known forcing term on entire domain */

typedef struct {
  char  *filename;
  Vector *sy_values;
} Type2;                       /* .pfb file */

typedef struct {
  char  *filename;
  Vector *sy_values;
} Type3;                       /* .nc file */


/*--------------------------------------------------------------------------
 * YSlope
 *--------------------------------------------------------------------------*/

void         YSlope(
                    ProblemData *problem_data,
                    Vector *     y_sl,
                    Vector *     dummy)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Grid             *grid3d = instance_xtra->grid3d;

  GrGeomSolid      *gr_solid, *gr_domain;

  Type0            *dummy0;
  Type1            *dummy1;
  Type2            *dummy2;
  Type3            *dummy3;

  SubgridArray     *subgrids = GridSubgrids(grid3d);

  VectorUpdateCommHandle       *handle;

  Subgrid          *subgrid;
  Subvector        *ps_sub;
  Subvector        *sy_values_sub;

  double           *data;
  double           *psdat, *sy_values_dat;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  int is, i, j, k, ips, ipicv;
  double time = 0.0;

  (void)dummy;

  /*-----------------------------------------------------------------------
   * Put in any user defined sources for this phase
   *-----------------------------------------------------------------------*/

  InitVectorAll(y_sl, 0.0);

  switch ((public_xtra->type))
  {
    case 0:
    {
      int num_regions;
      int     *region_indices;
      double  *values;

      double value;
      int ir;

      dummy0 = (Type0*)(public_xtra->data);

      num_regions = (dummy0->num_regions);
      region_indices = (dummy0->region_indices);
      values = (dummy0->values);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
        value = values[ir];

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          ps_sub = VectorSubvector(y_sl, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          data = SubvectorData(ps_sub);
          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ips = SubvectorEltIndex(ps_sub, i, j, 0);

            data[ips] = value;
          });
        }
      }

      break;
    }     /* End case 0 */

    case 1:
    {
      GrGeomSolid  *gr_domain;
      double x, y, z;
      int function_type;

      dummy1 = (Type1*)(public_xtra->data);

      function_type = (dummy1->function_type);

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        ps_sub = VectorSubvector(y_sl, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        /* RDF: assume resolution is the same in all 3 directions */
        r = SubgridRX(subgrid);

        data = SubvectorData(ps_sub);

        switch (function_type)
        {
          case 1:       /* p= x */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              x = RealSpaceX(i, SubgridRX(subgrid));
              /* nonlinear case -div(p grad p) = f */
              data[ips] = -1.0;
            });
            break;
          }        /* End case 1 */

          case 2:    /* topo = cos(x)/8 + sin(y)/20 + x^(1/8) + y^(1/8) + cos(x/5)/(5*8) + sin(y/5)/(5*10)
                      * Sy =  cos(y)/ 10 + (1/2)y^(-1/2) + cos(y/5)/(5*10) */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              x = RealSpaceX(i, SubgridRX(subgrid));
              y = RealSpaceY(j, SubgridRY(subgrid));
              /* nonlinear case -div(p grad p) = f */
              //data[ips] = cos(y)/20.0 + (1.0/8.0)*pow(x,-(7/8)) +cos(y/5.0)/(5.0*10.0);
              data[ips] = (0.10) * cos(y / 5.0);
            });
            break;
          }        /* End case 2 */

          case 3:       /* p= x^3y^2 + sinxy + 1 */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              x = RealSpaceX(i, SubgridRX(subgrid));
              y = RealSpaceY(j, SubgridRY(subgrid));
              /* nonlinear case -div(p grad p) = f */
              data[ips] = -pow((3 * x * x * y * y + y * cos(x * y)), 2) - pow((2 * x * x * x * y + x * cos(x * y)), 2) - (x * x * x * y * y + sin(x * y) + 1) * (6 * x * y * y + 2 * x * x * x - (x * x + y * y) * sin(x * y));
            });
            break;
          }        /* End case 3 */

          case 4:       /* f for p = x^3y^4 + x^2 + sinxy cosy + 1 */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              x = RealSpaceX(i, SubgridRX(subgrid));
              y = RealSpaceY(j, SubgridRY(subgrid));
              z = RealSpaceZ(k, SubgridRZ(subgrid));

              data[ips] = -pow(3 * x * x * pow(y, 4) + 2 * x + y * cos(x * y) * cos(y), 2) - pow(4 * x * x * x * y * y * y + x * cos(x * y) * cos(y) - sin(x * y) * sin(y), 2) - (x * x * x * pow(y, 4) + x * x + sin(x * y) * cos(y) + 1) * (6 * x * pow(y, 4) + 2 - (x * x + y * y + 1) * sin(x * y) * cos(y) + 12 * x * x * x * y * y - 2 * x * cos(x * y) * sin(y));
            });
            break;
          }        /* End case 4 */

          case 5:       /* f = xyz-y^2z^2t^2-x^2z^2t^2-x^2y^2t^2 (p=xyzt+1)*/
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              x = RealSpaceX(i, SubgridRX(subgrid));
              y = RealSpaceY(j, SubgridRY(subgrid));
              z = RealSpaceZ(k, SubgridRZ(subgrid));

              data[ips] = x * y * z - time * time * (y * y * z * z + x * x * z * z + x * x * y * y);
            });
            break;
          }        /* End case 5 */

          case 6:       /* f = xyz-y^2z^2t^2-2x^2z^2t^2-3x^2y^2t^2 (p=xyzt+1,
                         * K=(1; 2; 3) )*/
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              x = RealSpaceX(i, SubgridRX(subgrid));
              y = RealSpaceY(j, SubgridRY(subgrid));
              z = RealSpaceZ(k, SubgridRZ(subgrid));

              data[ips] = x * y * z
                          - time * time * (y * y * z * z + x * x * z * z * 2.0 + x * x * y * y * 3.0);
            });
            break;
          }        /* End case 6 */
        }       /* End switch statement on function_types */
      }        /* End subgrid loop */

      break;
    }     /* End case 1 for input types */

    case 2:
    {
      Vector *sy_val;

      dummy2 = (Type2*)(public_xtra->data);

      sy_val = dummy2->sy_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        ps_sub = VectorSubvector(y_sl, is);
        sy_values_sub = VectorSubvector(sy_val, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        r = SubgridRX(subgrid);

        psdat = SubvectorData(ps_sub);
        sy_values_dat = SubvectorData(sy_values_sub);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(ps_sub, i, j, 0);
          ipicv = SubvectorEltIndex(sy_values_sub, i, j, 0);

          psdat[ips] = sy_values_dat[ipicv];
        });
      }    /* End subgrid loop */

      break;
    }

    case 3:
    {
      Vector *sy_val;

      dummy3 = (Type3*)(public_xtra->data);

      sy_val = dummy3->sy_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        ps_sub = VectorSubvector(y_sl, is);
        sy_values_sub = VectorSubvector(sy_val, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        r = SubgridRX(subgrid);

        psdat = SubvectorData(ps_sub);
        sy_values_dat = SubvectorData(sy_values_sub);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(ps_sub, i, j, 0);
          ipicv = SubvectorEltIndex(sy_values_sub, i, j, 0);

          psdat[ips] = sy_values_dat[ipicv];
        });
      }    /* End subgrid loop */

      break;
    }
  }    /* End switch statement for input types */

  handle = InitVectorUpdate(y_sl, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
}


/*--------------------------------------------------------------------------
 * YSlopeInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *YSlopeInitInstanceXtra(
                                  Grid *grid3d,
                                  Grid *grid2d)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  Type2  *dummy2;
  Type3  *dummy3;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid3d != NULL)
  {
    (instance_xtra->grid3d) = grid3d;
  }

  if (grid2d != NULL)
  {
    (instance_xtra->grid2d) = grid2d;

    if (public_xtra->type == 2)
    {
      dummy2 = (Type2*)(public_xtra->data);

      dummy2->sy_values = NewVectorType(grid2d, 1, 1,
                                        vector_cell_centered_2D);

      ReadPFBinary((dummy2->filename), (dummy2->sy_values));
    }

    if (public_xtra->type == 3)
    {
      dummy3 = (Type3*)(public_xtra->data);

      dummy3->sy_values = NewVectorType(grid2d, 1, 1,
                                        vector_cell_centered_2D);

      ReadPFNC((dummy3->filename), (dummy3->sy_values), "slopey", 0, 2);
    }
  }


  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * YSlopeFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  YSlopeFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Type2  *dummy2;
  Type3  *dummy3;

  if (public_xtra->type == 2)
  {
    dummy2 = (Type2*)(public_xtra->data);
    FreeVector(dummy2->sy_values);
  }

  if (public_xtra->type == 3)
  {
    dummy3 = (Type3*)(public_xtra->data);
    FreeVector(dummy3->sy_values);
  }

  if (instance_xtra)
  {
    free(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * YSlopeNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *YSlopeNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;
  Type2         *dummy2;
  Type3         *dummy3;

  int num_regions, ir;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;

  NameArray type_na;
  NameArray function_type_na;

  type_na = NA_NewNameArray("Constant PredefinedFunction PFBFile NCFile");

  function_type_na = NA_NewNameArray("dum0 X SineCosTopo X3Y2PlusSinXYPlus1 \
                                       X3Y4PlusX2PlusSinXYCosYPlus1 \
                                       XYZTPlus1 XYZTPlus1PermTensor");
  public_xtra = ctalloc(PublicXtra, 1);

  switch_name = GetString("TopoSlopesY.Type");

  public_xtra->type = NA_NameToIndexExitOnError(type_na, switch_name, "TopoSlopesY.Type");

  switch ((public_xtra->type))
  {
    case 0:
    {
      dummy0 = ctalloc(Type0, 1);

      switch_name = GetString("TopoSlopesY.GeomNames");

      dummy0->regions = NA_NewNameArray(switch_name);

      num_regions = (dummy0->num_regions) = NA_Sizeof(dummy0->regions);

      (dummy0->region_indices) = ctalloc(int, num_regions);
      (dummy0->values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        dummy0->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames,
                         NA_IndexToName(dummy0->regions, ir));

        sprintf(key, "TopoSlopesY.Geom.%s.Value", NA_IndexToName(dummy0->regions, ir));
        dummy0->values[ir] = GetDouble(key);
      }

      (public_xtra->data) = (void*)dummy0;

      break;
    }     /* End case 0 */

    case 1:
    {
      dummy1 = ctalloc(Type1, 1);

      switch_name = GetString("TopoSlopesY.PredefinedFunction");

      dummy1->function_type =
        NA_NameToIndexExitOnError(function_type_na, switch_name, "TopoSlopesY.PredefinedFunction");

      (public_xtra->data) = (void*)dummy1;

      break;
    }     /* End case 1 */

    case 2:
    {
      dummy2 = ctalloc(Type2, 1);

      dummy2->filename = GetString("TopoSlopesY.FileName");

      (public_xtra->data) = (void*)dummy2;
      break;
    }

    case 3:
    {
      dummy3 = ctalloc(Type3, 1);

      dummy3->filename = GetString("TopoSlopesY.FileName");

      (public_xtra->data) = (void*)dummy3;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }    /* End case statement */


  NA_FreeNameArray(type_na);
  NA_FreeNameArray(function_type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * YSlopeFreePublicXtra
 *-------------------------------------------------------------------------*/

void  YSlopeFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;
  Type1       *dummy1;


  if (public_xtra)
  {
    switch ((public_xtra->type))
    {
      case 0:
      {
        dummy0 = (Type0*)(public_xtra->data);

        NA_FreeNameArray(dummy0->regions);

        tfree(dummy0->region_indices);
        tfree(dummy0->values);
        tfree(dummy0);
        break;
      }

      case 1:
      {
        dummy1 = (Type1*)(public_xtra->data);

        tfree(dummy1);
        break;
      }

      case 2:
      {
        Type2 * dummy2;
        dummy2 = (Type2*)(public_xtra->data);

        tfree(dummy2);
        break;
      }

      case 3:
      {
        Type3 * dummy3;
        dummy3 = (Type3*)(public_xtra->data);

        tfree(dummy3);
        break;
      }
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * YSlopeSizeOfTempData
 *--------------------------------------------------------------------------*/

int  YSlopeSizeOfTempData()
{
  return 0;
}
