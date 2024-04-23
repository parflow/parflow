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

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;
  int num_contaminants;

  int     *type;
  void   **data;
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  int num_regions;
  NameArray regions;

  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
  char  *filename;
} Type1;                      /* .pfb file */


/*--------------------------------------------------------------------------
 * ICPhaseConcen
 *--------------------------------------------------------------------------*/

void         ICPhaseConcen(
                           Vector *     ic_phase_concen,
                           int          phase,
                           int          contaminant,
                           ProblemData *problem_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Grid           *grid = VectorGrid(ic_phase_concen);

  Type0          *dummy0;
  Type1          *dummy1;

  SubgridArray   *subgrids = GridSubgrids(grid);

  Subgrid        *subgrid;
  Subvector      *pc_sub;

  int index;

  double         *data;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  double field_sum;

  int is, i, j, k, ipc;


  /*-----------------------------------------------------------------------
   * Initial conditions for concentrations in each phase
   *-----------------------------------------------------------------------*/

  InitVector(ic_phase_concen, 0.0);

  index = phase * (public_xtra->num_contaminants) + contaminant;

  switch ((public_xtra->type[index]))
  {
    case 0:
    {
      int num_regions;
      int     *region_indices;
      double  *values;

      GrGeomSolid  *gr_solid;
      double value;
      int ir;


      dummy0 = (Type0*)(public_xtra->data[index]);

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
          pc_sub = VectorSubvector(ic_phase_concen, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          data = SubvectorData(pc_sub);
          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ipc = SubvectorEltIndex(pc_sub, i, j, k);

            data[ipc] = value;
          });
        }
      }

      break;
    }

    case 1:
    {
      dummy1 = (Type1*)(public_xtra->data[index]);

      ReadPFBinary((dummy1->filename), ic_phase_concen);

      break;
    }
  }

#if 1
  {
    /*************************************************************************
    *             Informational computation and printing.                   *
    *************************************************************************/

    field_sum = ComputeTotalConcen(ProblemDataGrDomain(problem_data),
                                   grid, ic_phase_concen);

    if (!amps_Rank(amps_CommWorld))
    {
      amps_Printf("Initial concentration volume for phase %2d, contaminant %3d = %f\n", phase, contaminant, field_sum);
    }
  }
#endif
}

/*--------------------------------------------------------------------------
 * ICPhaseConcenInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *ICPhaseConcenInitInstanceXtra()
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

/*-------------------------------------------------------------------------
 * ICPhaseConcenFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  ICPhaseConcenFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * ICPhaseConcenNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ICPhaseConcenNewPublicXtra(
                                       int num_phases,
                                       int num_contaminants)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;

  int i;
  int phase;
  int cont;

  char key[IDB_MAX_KEY_LEN];
  char *switch_name;

  NameArray type_na;

  type_na = NA_NewNameArray("Constant PFBFile");

  if (num_contaminants > 0)
  {
    public_xtra = ctalloc(PublicXtra, 1);

    (public_xtra->num_phases) = num_phases;
    (public_xtra->num_contaminants) = num_contaminants;

    (public_xtra->type) = ctalloc(int, num_phases * num_contaminants);
    (public_xtra->data) = ctalloc(void *, num_phases * num_contaminants);

    for (phase = 0; phase < num_phases; phase++)
    {
      for (cont = 0; cont < num_contaminants; cont++)
      {
        i = phase + cont;

        sprintf(key, "PhaseConcen.%s.%s.Type",
                NA_IndexToName(GlobalsPhaseNames, phase),
                NA_IndexToName(GlobalsContaminatNames, cont));

        switch_name = GetString(key);

        public_xtra->type[i] = NA_NameToIndexExitOnError(type_na, switch_name, key);

        switch ((public_xtra->type[i]))
        {
          case 0:
          {
            int num_regions, ir;

            dummy0 = ctalloc(Type0, 1);

            sprintf(key, "PhaseConcen.%s.%s.GeomNames",
                    NA_IndexToName(GlobalsPhaseNames, phase),
                    NA_IndexToName(GlobalsContaminatNames, cont));
            switch_name = GetString(key);

            dummy0->regions = NA_NewNameArray(switch_name);

            num_regions =
              (dummy0->num_regions) = NA_Sizeof(dummy0->regions);

            (dummy0->region_indices) = ctalloc(int, num_regions);
            (dummy0->values) = ctalloc(double, num_regions);

            for (ir = 0; ir < num_regions; ir++)
            {
              dummy0->region_indices[ir] =
                NA_NameToIndex(GlobalsGeomNames,
                               NA_IndexToName(dummy0->regions, ir));

              sprintf(key, "PhaseConcen.%s.%s.Geom.%s.Value",
                      NA_IndexToName(GlobalsPhaseNames, phase),
                      NA_IndexToName(GlobalsContaminatNames, cont),
                      NA_IndexToName(dummy0->regions, ir));
              dummy0->values[ir] = GetDouble(key);
            }

            (public_xtra->data[i]) = (void*)dummy0;

            break;
          }

          case 1:
          {
            dummy1 = ctalloc(Type1, 1);


            sprintf(key, "PhaseConcen.%s.%s.FileName",
                    NA_IndexToName(GlobalsPhaseNames, phase),
                    NA_IndexToName(GlobalsContaminatNames, cont));

            dummy1->filename = GetString(key);

            (public_xtra->data[i]) = (void*)dummy1;

            break;
          }

          default:
          {
            InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
          }
        }
      }
    }
  }
  else
  {
    public_xtra = NULL;
  }

  NA_FreeNameArray(type_na);

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}

/*--------------------------------------------------------------------------
 * ICPhaseConcenFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ICPhaseConcenFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  int num_phases;
  int num_contaminants;

  Type0       *dummy0;
  Type1       *dummy1;

  int i;


  if (public_xtra)
  {
    num_phases = (public_xtra->num_phases);
    num_contaminants = (public_xtra->num_contaminants);

    for (i = 0; i < num_phases * num_contaminants; i++)
    {
      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = (Type0*)(public_xtra->data[i]);

          NA_FreeNameArray(dummy0->regions);

          tfree(dummy0->region_indices);
          tfree(dummy0->values);
          tfree(dummy0);
          break;
        }

        case 1:
        {
          dummy1 = (Type1*)(public_xtra->data[i]);

          tfree(dummy1);
          break;
        }
      }
    }

    tfree(public_xtra->data);
    tfree(public_xtra->type);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * ICPhaseConcenSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ICPhaseConcenSizeOfTempData()
{
  return 0;
}
