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


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;

  int num_patches;
  NameArray patches;

  int    *patch_indexes;  /* num_patches patch indexes */
  int    *input_types;    /* num_patches input types */
  int    *bc_types;       /* num_patches BC types */
  void  **data;           /* num_patches pointers to Type structures */
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  double constant;
} Type0;               /* Dirichlet, constant */

typedef struct {
  double height;        /* height */
  double lower;         /* value below height */
  double upper;         /* value above height */
} Type1;               /* Dirichlet, constant saturation line height */

typedef struct {
  double xlower, ylower, xupper, yupper;
  int num_points;
  double  *point;      /* num_points points */
  double  *height;     /* num_points heights */
  double lower;        /* value below height */
  double upper;        /* value above height */
} Type2;               /* Dirichlet, piecewise linear saturation line height */


/*--------------------------------------------------------------------------
 * BCPhaseSaturation
 *   This routine implements the saturation boundary conditions
 *   (Dirichlet only) by setting the saturations of 3 ghost layers
 *   outside of the boundary.
 *--------------------------------------------------------------------------*/

void          BCPhaseSaturation(
                                Vector *     saturation,
                                int          phase,
                                GrGeomSolid *gr_domain)
{
  PFModule       *this_module = ThisPFModule;
  PublicXtra     *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0          *dummy0;
  Type1          *dummy1;
  Type2          *dummy2;

  int num_patches = (public_xtra->num_patches);
  int            *patch_indexes = (public_xtra->patch_indexes);
  int            *input_types = (public_xtra->input_types);
  int            *bc_types = (public_xtra->bc_types);

  Grid           *grid = VectorGrid(saturation);
  SubgridArray   *subgrids = GridSubgrids(grid);

  Subgrid        *subgrid;

  Subvector      *sat_sub;
  double         *satp;

  BCStruct       *bc_struct;

  int nx_v, ny_v;
  int sx_v, sy_v, sz_v;

  int indx, ipatch, is, i, j, k, ival=0;
  PF_UNUSED(ival);

  /*-----------------------------------------------------------------------
   * Get an offset into the PublicXtra data
   *-----------------------------------------------------------------------*/

  indx = (phase * num_patches);

  /*-----------------------------------------------------------------------
   * Set up bc_struct with NULL values component
   *-----------------------------------------------------------------------*/

  bc_struct = NewBCStruct(subgrids, gr_domain,
                          num_patches, patch_indexes, bc_types, NULL);

  /*-----------------------------------------------------------------------
   * Implement BC's
   *-----------------------------------------------------------------------*/

  for (ipatch = 0; ipatch < num_patches; ipatch++)
  {
    ForSubgridI(is, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, is);


      sat_sub = VectorSubvector(saturation, is);

      nx_v = SubvectorNX(sat_sub);
      ny_v = SubvectorNY(sat_sub);

      sx_v = 1;
      sy_v = nx_v;
      sz_v = ny_v * nx_v;

      satp = SubvectorData(sat_sub);

      switch (input_types[indx + ipatch])
      {
        case 0:
        {
          double constant;
          ForPatchCellsPerFace(BC_ALL,
                               BeforeAllCells({
                                   dummy0 = (Type0*)(public_xtra->data[indx + ipatch]);
                                   constant = (dummy0->constant);
                                 }),
                               LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                               Locals(int sv, iv;),
                               CellSetup({
                                   sv = 0;
                                   iv = SubvectorEltIndex(sat_sub, i, j, k);
                                 }),
                               FACE(LeftFace,  { sv = -sx_v; }),
                               FACE(RightFace, { sv = sx_v; }),
                               FACE(DownFace,  { sv = -sy_v; }),
                               FACE(UpFace,    { sv = sy_v; }),
                               FACE(BackFace,  { sv = -sz_v; }),
                               FACE(FrontFace, { sv = sz_v; }),
                               CellFinalize({
                                   satp[iv]          = constant;
                                   satp[iv + sv]     = constant;
                                   satp[iv + 2 * sv] = constant;
                                 }),
                               AfterAllCells(DoNothing)
            );
          break;
        }

        case 1:
        {
          double height, lower, upper, dz2;

          ForPatchCellsPerFace(BC_ALL,
                               BeforeAllCells({
                                   dummy1 = (Type1*)(public_xtra->data[indx + ipatch]);
                                   height = (dummy1->height);
                                   lower = (dummy1->lower);
                                   upper = (dummy1->upper);
                                   dz2 = SubgridDZ(subgrid) / 2.0;
                                 }),
                               LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                               Locals(int sv, iv; double z;),
                               CellSetup({
                                   sv = 0;
                                   z = RealSpaceZ(k, SubgridRZ(subgrid));
                                   iv = SubvectorEltIndex(sat_sub, i, j, k);
                                 }),
                               FACE(LeftFace,  { sv = -sx_v; }),
                               FACE(RightFace, { sv = sx_v; }),
                               FACE(DownFace,  { sv = -sy_v; }),
                               FACE(UpFace,    { sv = sy_v; }),
                               FACE(BackFace, {
                                   sv = -sz_v;
                                   z = z - dz2;
                                 }),
                               FACE(FrontFace, {
                                   sv = sz_v;
                                   z = z + dz2;
                                 }),
                               CellFinalize({
                                   if (z <= height)
                                   {
                                     satp[iv       ] = lower;
                                     satp[iv + sv] = lower;
                                     satp[iv + 2 * sv] = lower;
                                   }
                                   else
                                   {
                                     satp[iv       ] = upper;
                                     satp[iv + sv] = upper;
                                     satp[iv + 2 * sv] = upper;
                                   }
                                 }),
                               AfterAllCells(DoNothing)
            );
          break;
        }

        case 2:
        {
          int num_points;
          double  *point;
          double  *height;
          double lower;
          double upper;

          double dx2, dy2, dz2;
          double unitx, unity, line_min, line_length;

          ForPatchCellsPerFace(BC_ALL,
                               BeforeAllCells({
                                   dummy2 = (Type2*)(public_xtra->data[indx + ipatch]);
                                   num_points = (dummy2->num_points);
                                   point = (dummy2->point);
                                   height = (dummy2->height);
                                   lower = (dummy2->lower);
                                   upper = (dummy2->upper);

                                   dx2 = SubgridDX(subgrid) / 2.0;
                                   dy2 = SubgridDY(subgrid) / 2.0;
                                   dz2 = SubgridDZ(subgrid) / 2.0;

                                   /* compute unit direction vector for piecewise linear line */
                                   unitx = (dummy2->xupper) - (dummy2->xlower);
                                   unity = (dummy2->yupper) - (dummy2->ylower);
                                   line_length = sqrt(unitx * unitx + unity * unity);
                                   unitx /= line_length;
                                   unity /= line_length;
                                   line_min = (dummy2->xlower) * unitx
                                              + (dummy2->ylower) * unity;
                                 }),
                               LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                               Locals(int sv, iv, ip;
                                      double x, y, z, xy, slope, interp_height;),
                               CellSetup({
                                   sv = 0;
                                   x = RealSpaceX(i, SubgridRX(subgrid));
                                   y = RealSpaceY(j, SubgridRY(subgrid));
                                   z = RealSpaceZ(k, SubgridRZ(subgrid));
                                   iv = SubvectorEltIndex(sat_sub, i, j, k);
                                 }),
                               FACE(LeftFace, {
                                   sv = -sx_v;
                                   x = x - dx2;
                                 }),
                               FACE(RightFace, {
                                   sv = sx_v;
                                   x = x + dx2;
                                 }),
                               FACE(DownFace, {
                                   sv = -sy_v;
                                   y = y - dy2;
                                 }),
                               FACE(UpFace, {
                                   sv = sy_v;
                                   y = y + dy2;
                                 }),
                               FACE(BackFace, {
                                   sv = -sz_v;
                                   z = z - dz2;
                                 }),
                               FACE(FrontFace, {
                                   sv = sz_v;
                                   z = z + dz2;
                                 }),
                               CellFinalize(
                               {
                                 /* project center of BC face onto piecewise linear line */
                                 xy = x * unitx + y * unity;
                                 xy = (xy - line_min) / line_length;

                                 /* find two neighboring points */
                                 ip = 1;
                                 for (; ip < (num_points - 1); ip++)
                                 {
                                   if (xy < point[ip])
                                     break;
                                 }

                                 /* compute the slope */
                                 slope = ((height[ip] - height[ip - 1]) /
                                          (point[ip] - point[ip - 1]));

                                 interp_height = height[ip - 1] + slope * (xy - point[ip - 1]);

                                 if (z <= interp_height)
                                 {
                                   satp[iv       ] = lower;
                                   satp[iv + sv] = lower;
                                   satp[iv + 2 * sv] = lower;
                                 }
                                 else
                                 {
                                   satp[iv       ] = upper;
                                   satp[iv + sv] = upper;
                                   satp[iv + 2 * sv] = upper;
                                 }
                               }),
                               AfterAllCells(DoNothing)
            );

          break;
        }
      }
    }
  }

  /*-----------------------------------------------------------------------
   * Free up space
   *-----------------------------------------------------------------------*/

  FreeBCStruct(bc_struct);
}


/*--------------------------------------------------------------------------
 * BCPhaseSaturationInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *BCPhaseSaturationInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


#if 0
  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
#endif
  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * BCPhaseSaturationFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void BCPhaseSaturationFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * BCPhaseSaturationNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *BCPhaseSaturationNewPublicXtra(
                                          int num_phases)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  int num_patches = 0;

  Type0          *dummy0;
  Type1          *dummy1;
  Type2          *dummy2;

  int i, j, indx, size;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;
  int domain_index;
  char *phase_name;
  char *patch_name;
  char *patches;

  NameArray type_na;

  type_na = NA_NewNameArray("DirConstant ConstantWTHeight PLinearWTHeight");

  /* allocate space for the public_xtra structure */
  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->num_phases) = num_phases;

  if (num_phases > 1)
  {
    patches = GetString("BCSaturation.PatchNames");

    public_xtra->patches = NA_NewNameArray(patches);


    public_xtra->num_patches =
      num_patches = NA_Sizeof(public_xtra->patches);

    size = (num_phases - 1) * num_patches;

    (public_xtra->patch_indexes) = ctalloc(int, size);
    (public_xtra->input_types) = ctalloc(int, size);
    (public_xtra->bc_types) = ctalloc(int, size);
    (public_xtra->data) = ctalloc(void *, size);
  }

  /* Determine the domain geom index from domain name */
  switch_name = GetString("Domain.GeomName");
  domain_index = NA_NameToIndexExitOnError(GlobalsGeomNames, switch_name, "Domain.GeomName");

  indx = 0;
  for (i = 0; i < num_phases - 1; i++)
  {
    phase_name = NA_IndexToName(GlobalsPhaseNames, i);

    for (j = 0; j < num_patches; j++)
    {
      patch_name = NA_IndexToName(public_xtra->patches, i);

      public_xtra->patch_indexes[indx] =
        NA_NameToIndex(GlobalsGeometries[domain_index]->patches,
		       patch_name);

      if(public_xtra->patch_indexes[indx] < 0)
      {
	NA_InputError(GlobalsGeometries[domain_index]->patches, patch_name, "");
      }

      sprintf(key, "Patch.%s.BCSaturation.%s.Type", patch_name, phase_name);
      switch_name = GetString(key);
      public_xtra->input_types[indx] =
        NA_NameToIndexExitOnError(type_na, switch_name, key);

      switch ((public_xtra->input_types[indx]))
      {
        case 0:
        {
          (public_xtra->bc_types[indx]) = DirichletBC;

          dummy0 = ctalloc(Type0, 1);

          sprintf(key, "Patch.%s.BCSaturation.%s.Value",
                  patch_name, phase_name);
          dummy0->constant = GetDouble(key);

          (public_xtra->data[indx]) = (void*)dummy0;
          break;
        }

        case 1:
        {
          (public_xtra->bc_types[indx]) = DirichletBC;

          dummy1 = ctalloc(Type1, 1);

          sprintf(key, "Patch.%s.BCSaturation.%s.Value",
                  patch_name, phase_name);
          dummy1->height = GetDouble(key);

          (dummy1->lower) = 1.0;
          (dummy1->upper) = 0.0;

          (public_xtra->data[indx]) = (void*)dummy1;
          break;
        }

        case 2:
        {
          int k, num_points;

          (public_xtra->bc_types[indx]) = DirichletBC;

          dummy2 = ctalloc(Type2, 1);

          sprintf(key, "Patch.%s.BCSaturation.%s.XLower",
                  patch_name, phase_name);
          dummy2->xlower = GetDouble(key);

          sprintf(key, "Patch.%s.BCSaturation.%s.YLower",
                  patch_name, phase_name);
          dummy2->ylower = GetDouble(key);

          sprintf(key, "Patch.%s.BCSaturation.%s.XUpper",
                  patch_name, phase_name);
          dummy2->xupper = GetDouble(key);

          sprintf(key, "Patch.%s.BCSaturation.%s.YUpper",
                  patch_name, phase_name);
          dummy2->yupper = GetDouble(key);

          sprintf(key, "Patch.%s.BCPressure.%s.NumPoints", patch_name,
                  phase_name);
          num_points = dummy2->num_points = GetInt(key);

          (dummy2->point) = ctalloc(double, (dummy2->num_points));
          (dummy2->height) = ctalloc(double, (dummy2->num_points));

          for (k = 0; k < num_points; k++)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.%d.Location",
                    patch_name, phase_name, k);
            dummy2->point[k] = GetDouble(key);

            sprintf(key, "Patch.%s.BCPressure.%s.%d.Value",
                    patch_name, phase_name, k);
            dummy2->height[k] = GetDouble(key);
          }

          (dummy2->lower) = 1.0;
          (dummy2->upper) = 0.0;

          (public_xtra->data[indx]) = (void*)dummy2;
          break;
        }

        default:
        {
	  InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
        }
      }
      indx++;
    }
  }

  NA_FreeNameArray(type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * BCPhaseSaturationFreePublicXtra
 *--------------------------------------------------------------------------*/

void  BCPhaseSaturationFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  int num_phases = (public_xtra->num_phases);

  Type0  *dummy0;
  Type1  *dummy1;
  Type2  *dummy2;

  int i, j, indx;

  if (public_xtra)
  {
    indx = 0;

    NA_FreeNameArray(public_xtra->patches);

    for (i = 0; i < num_phases - 1; i++)
    {
      for (j = 0; j < (public_xtra->num_patches); j++)
      {
        switch ((public_xtra->input_types[indx]))
        {
          case 0:
            dummy0 = (Type0*)(public_xtra->data[indx]);
            tfree(dummy0);
            break;

          case 1:
            dummy1 = (Type1*)(public_xtra->data[indx]);
            tfree(dummy1);
            break;

          case 2:
            dummy2 = (Type2*)(public_xtra->data[indx]);
            tfree(dummy2->point);
            tfree(dummy2->height);
            tfree(dummy2);
            break;
        }
        indx++;
      }
    }

    tfree(public_xtra->data);
    tfree(public_xtra->bc_types);
    tfree(public_xtra->input_types);
    tfree(public_xtra->patch_indexes);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * BCPhaseSaturationSizeOfTempData
 *--------------------------------------------------------------------------*/

int  BCPhaseSaturationSizeOfTempData()
{
  return 0;
}
