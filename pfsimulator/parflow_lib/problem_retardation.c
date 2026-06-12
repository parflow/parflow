/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
  int num_contaminants;

  int num_geo_unit_indexes;
  int    *geo_indexes;
  NameArray geo_indexes_na;

  int    *type;
  int    *contam_retard_flag;
  void  **data;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  double  *temp_data;
  Vector *trans_x_velocity;
  Vector *trans_y_velocity;
  Vector *trans_z_velocity;

  Grid *grid;
} InstanceXtra;

typedef struct {
  double  *value;
} Type0;                       /* linear retardation */


/*--------------------------------------------------------------------------
 * Retardation
 * Originally this subroutine produced a lumped retardation/porosity vector of
 * the type:
 * solidmassfactor = porosity + (1 - porosity) * Kd
 * where Kd is the user supplied retardation rate.
 *
 * The changes made to the transport routine necessitate the separation of
 * porosity and retardation. This change allows the transport scheme to simulate
 * transient saturation and porosity effects (time-variable volume) independently
 * of the retardation of a chemical species.
 *
 * The retardation factor now takes the form:
 * retardation = 1 + (1 - porosity) * (Kd/porosity)
 * This retardation factor relates to the previous method:
 * retardation = solidmassfactor / porosity
 *
 * For each chemical component that is retarded, the vector of retardation values
 * is applied directly to the velocity field via upwinding of the retardation
 * factor:  transport_velocity = darcy_velocity / upstream_retardation     JJB
 *--------------------------------------------------------------------------*/

void         Retardation(Vector *     retard_vector,
                         Vector *     x_velocity,
                         Vector *     y_velocity,
                         Vector *     z_velocity,
                         Vector **    trans_x_vel_ptr,
                         Vector **    trans_y_vel_ptr,
                         Vector **    trans_z_vel_ptr,
                         int          contaminant,
                         ProblemData *problem_data)
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  GrGeomSolid **gr_solids = ProblemDataGrSolids(problem_data);
  Vector       *porosity = ProblemDataPorosity(problem_data);
  Grid         *grid = instance_xtra->grid;

  Type0        *dummy0;
  GrGeomSolid *gr_solid;

  Subgrid      *subgrid;
  Subvector    *r_sub, *p_sub;

  VectorUpdateCommHandle     *handle;

  int is, ig;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  double *rp, *pp;

  int i, j, k;
  int index;
  int ir, ip;

  /*-----------------------------------------------------------------------
   * Put in any user defined sources for this phase
   *-----------------------------------------------------------------------*/
  if (public_xtra->contam_retard_flag[contaminant])
  {
    InitVectorAll(retard_vector, 1.0);

    for (ig = 0; ig < (public_xtra->num_geo_unit_indexes); ig++)
    {
      gr_solid = gr_solids[(public_xtra->geo_indexes)[ig]];
      index = (public_xtra->num_contaminants) * ig + contaminant;

      switch ((public_xtra->type[index]))
      {
        case 0:
        {
          double value;

          dummy0 = (Type0*)(public_xtra->data[index]);

          value = (dummy0->value)[0];

          for (is = 0; is < GridNumSubgrids(grid); is++)
          {
            subgrid = GridSubgrid(grid, is);
            r_sub = VectorSubvector(retard_vector, is);
            p_sub = VectorSubvector(porosity, is);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            /* RDF: assume resolution is the same in all 3 directions */
            r = SubgridRX(subgrid);

            rp = SubvectorData(r_sub);
            pp = SubvectorData(p_sub);
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              ir = SubvectorEltIndex(r_sub, i, j, k);
              ip = SubvectorEltIndex(p_sub, i, j, k);

              rp[ir] = 1.0 + (1.0 - pp[ip]) * (value / pp[ip]);
            });
          }

          break;
        }
      }
    }

    handle = InitVectorUpdate(retard_vector, VectorUpdateAll2);
    FinalizeVectorUpdate(handle);

    TransportVelocity(x_velocity, y_velocity, z_velocity,
                      trans_x_vel_ptr, trans_y_vel_ptr, trans_z_vel_ptr,
                      retard_vector);
  }
  else // using Alquimia or no retardation for this component
  {
    *trans_x_vel_ptr = x_velocity;
    *trans_y_vel_ptr = y_velocity;
    *trans_z_vel_ptr = z_velocity;
  }
}



void  TransportVelocity(Vector * x_velocity,
                        Vector * y_velocity,
                        Vector * z_velocity,
                        Vector **trans_x_vel_ptr,
                        Vector **trans_y_vel_ptr,
                        Vector **trans_z_vel_ptr,
                        Vector * retard)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Grid         *grid;
  Subgrid      *subgrid;

  Subvector    *v_sub;
  Subvector    *tv_sub;
  Subvector    *r_sub;

  Vector       *velocity = NULL;
  Vector       *trans_vel = NULL;

  double       *vp, *tvp;
  double       *rlp, *rrp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int nx_r, ny_r, nz_r;
  int stx = 0, sty = 0, stz = 0;
  int i_s, dir, i, j, k, vi, ri;

  VectorUpdateCommHandle *handle;

  for (dir = 0; dir < 3; dir++)
  {
    switch (dir)
    {
      case 0:
        velocity = x_velocity;
        trans_vel = instance_xtra->trans_x_velocity;
        stx = 1;
        sty = 0;
        stz = 0;
        break;

      case 1:
        velocity = y_velocity;
        trans_vel = instance_xtra->trans_y_velocity;
        stx = 0;
        sty = 1;
        stz = 0;
        break;

      case 2:
        velocity = z_velocity;
        trans_vel = instance_xtra->trans_z_velocity;
        stx = 0;
        sty = 0;
        stz = 1;
        break;
    }

    grid = VectorGrid(velocity);
    ForSubgridI(i_s, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, i_s);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      v_sub = VectorSubvector(velocity, i_s);
      tv_sub = VectorSubvector(trans_vel, i_s);
      r_sub = VectorSubvector(retard, i_s);

      nx_v = SubvectorNX(v_sub);
      ny_v = SubvectorNY(v_sub);
      nz_v = SubvectorNZ(v_sub);

      nx_r = SubvectorNX(r_sub);
      ny_r = SubvectorNY(r_sub);
      nz_r = SubvectorNZ(r_sub);

      vp = SubvectorElt(v_sub, ix, iy, iz);
      tvp = SubvectorElt(tv_sub, ix, iy, iz);
      rlp = SubvectorElt(r_sub, ix - stx, iy - sty, iz - stz);
      rrp = SubvectorElt(r_sub, ix, iy, iz);

      vi = 0;
      ri = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                vi, nx_v, ny_v, nz_v, 1, 1, 1,
                ri, nx_r, ny_r, nz_r, 1, 1, 1,
      {
        if (vp[vi] > 0.0)
        {
          tvp[vi] = vp[vi] / rlp[ri];
        }
        else
        {
          tvp[vi] = vp[vi] / rrp[ri];
        }
      });
    }
  }

  handle = InitVectorUpdate(instance_xtra->trans_x_velocity, VectorUpdateVelX);
  FinalizeVectorUpdate(handle);

  handle = InitVectorUpdate(instance_xtra->trans_y_velocity, VectorUpdateVelY);
  FinalizeVectorUpdate(handle);

  handle = InitVectorUpdate(instance_xtra->trans_z_velocity, VectorUpdateVelZ);
  FinalizeVectorUpdate(handle);

  *trans_x_vel_ptr = instance_xtra->trans_x_velocity;
  *trans_y_vel_ptr = instance_xtra->trans_y_velocity;
  *trans_z_vel_ptr = instance_xtra->trans_z_velocity;
}




/*--------------------------------------------------------------------------
 * RetardationInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *RetardationInitInstanceXtra(
                                       Grid *grid, Grid *x_grid, Grid *y_grid,
                                       Grid *z_grid, double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      FreeVector(instance_xtra->trans_x_velocity);
      FreeVector(instance_xtra->trans_y_velocity);
      FreeVector(instance_xtra->trans_z_velocity);
    }

    (instance_xtra->grid) = grid;

    instance_xtra->trans_x_velocity =
      NewVectorType(x_grid, 1, 2, vector_side_centered_x);
    instance_xtra->trans_y_velocity =
      NewVectorType(y_grid, 1, 2, vector_side_centered_y);
    instance_xtra->trans_z_velocity =
      NewVectorType(z_grid, 1, 2, vector_side_centered_z);
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * RetardationFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  RetardationFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  FreeVector(instance_xtra->trans_x_velocity);
  FreeVector(instance_xtra->trans_y_velocity);
  FreeVector(instance_xtra->trans_z_velocity);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * RetardationNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *RetardationNewPublicXtra(
                                    int num_contaminants)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;

  int num_geo_unit_indexes;
  int i, ig;
  int ind, index;


  char          *geo_index_names;
  NameArray geo_index_na;

  char          *switch_name;
  char key[IDB_MAX_KEY_LEN];

  NameArray switch_na;

  char *geom_name;

  /*----------------------------------------------------------
   * The name array to map names to switch values
   *----------------------------------------------------------*/
  switch_na = NA_NewNameArray("Linear");




  if (GlobalsChemistryFlag)
  {
    if (num_contaminants > 0)
    {
      public_xtra = ctalloc(PublicXtra, 1);

      (public_xtra->num_contaminants) = num_contaminants;

      geo_index_names = GetString("Geom.Retardation.GeomNames");
      geo_index_na = (public_xtra->geo_indexes_na) =
        NA_NewNameArray(geo_index_names);
      num_geo_unit_indexes = NA_Sizeof(geo_index_na);

      (public_xtra->num_geo_unit_indexes) = num_geo_unit_indexes;
      (public_xtra->geo_indexes) = ctalloc(int, num_geo_unit_indexes);

      (public_xtra->type) = ctalloc(int, num_geo_unit_indexes * num_contaminants);
      (public_xtra->data) = ctalloc(void *, num_geo_unit_indexes * num_contaminants);

      (public_xtra->contam_retard_flag) = ctalloc(int, num_contaminants);


      for (ig = 0; ig < num_geo_unit_indexes; ig++)
      {
        geom_name = NA_IndexToName(geo_index_na, ig);

        ind = NA_NameToIndex(GlobalsGeomNames, geom_name);

        (public_xtra->geo_indexes)[ig] = ind;

        for (i = 0; i < num_contaminants; i++)
        {
          index = num_contaminants * ig + i;
          public_xtra->type[index] = 0;
          dummy0 = ctalloc(Type0, 1);
          (dummy0->value) = ctalloc(double, 1);
          *(dummy0->value) = 0.0;
          (public_xtra->data[index]) = (void*)dummy0;
        }
      }
    }
    else
    {
      public_xtra = NULL;
    }
  }
  else
  {
    if (num_contaminants > 0)
    {
      public_xtra = ctalloc(PublicXtra, 1);

      (public_xtra->num_contaminants) = num_contaminants;

      geo_index_names = GetString("Geom.Retardation.GeomNames");
      geo_index_na = (public_xtra->geo_indexes_na) =
        NA_NewNameArray(geo_index_names);
      num_geo_unit_indexes = NA_Sizeof(geo_index_na);

      (public_xtra->num_geo_unit_indexes) = num_geo_unit_indexes;
      (public_xtra->geo_indexes) = ctalloc(int, num_geo_unit_indexes);

      (public_xtra->type) = ctalloc(int, num_geo_unit_indexes * num_contaminants);
      (public_xtra->data) = ctalloc(void *, num_geo_unit_indexes * num_contaminants);

      (public_xtra->contam_retard_flag) = ctalloc(int, num_contaminants);

      for (ig = 0; ig < num_geo_unit_indexes; ig++)
      {
        geom_name = NA_IndexToName(geo_index_na, ig);

        ind = NA_NameToIndex(GlobalsGeomNames, geom_name);

        (public_xtra->geo_indexes)[ig] = ind;

        for (i = 0; i < num_contaminants; i++)
        {
          index = num_contaminants * ig + i;

          sprintf(key, "Geom.%s.%s.Retardation.Type",
                  geom_name,
                  NA_IndexToName(GlobalsContaminatNames, i));
          switch_name = GetString(key);

          public_xtra->type[index] =
            NA_NameToIndexExitOnError(switch_na, switch_name, key);

          switch ((public_xtra->type[index]))
          {
            case 0:
            {
              dummy0 = ctalloc(Type0, 1);

              (dummy0->value) = ctalloc(double, 1);

              sprintf(key, "Geom.%s.%s.Retardation.Rate",
                      geom_name,
                      NA_IndexToName(GlobalsContaminatNames, i));
              *(dummy0->value) = GetDoubleDefault(key, 0.0);

              if (*(dummy0->value) != 0.0)
              {
                (public_xtra->contam_retard_flag[i]) = 1;
              }

              (public_xtra->data[index]) = (void*)dummy0;

              break;
            }

            default:
            {
              InputError("Error: invalid retardation type <%s> for key <%s>\n",
                         switch_name, key);

              break;
            }
          }
        }
      }
    }
    else
    {
      public_xtra = NULL;
    }
  }

  NA_FreeNameArray(switch_na);
  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*-------------------------------------------------------------------------
 * RetardationFreePublicXtra
 *-------------------------------------------------------------------------*/

void  RetardationFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;

  int i, ig;
  int index;


  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->geo_indexes_na);

    for (ig = 0; ig < (public_xtra->num_geo_unit_indexes); ig++)
    {
      for (i = 0; i < (public_xtra->num_contaminants); i++)
      {
        index = (public_xtra->num_contaminants) * ig + i;

        switch ((public_xtra->type[index]))
        {
          case 0:
          {
            dummy0 = (Type0*)(public_xtra->data[index]);

            tfree(dummy0->value);

            tfree(dummy0);

            break;
          }
        }
      }
    }

    tfree(public_xtra->geo_indexes);

    tfree(public_xtra->data);
    tfree(public_xtra->type);
    tfree(public_xtra->contam_retard_flag);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * RetardationSizeOfTempData
 *--------------------------------------------------------------------------*/

int  RetardationSizeOfTempData()
{
  int sz = 0;

  return sz;
}

