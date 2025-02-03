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
* Member functions for the Geometry class.
*
*****************************************************************************/

#include "parflow.h"
#include "geometry.h"


/*--------------------------------------------------------------------------
 * GeomNewVertexArray
 *--------------------------------------------------------------------------*/

GeomVertexArray  *GeomNewVertexArray(
                                     GeomVertex **vertices,
                                     int          nV)
{
  GeomVertexArray   *new_geom_vertex_array;


  new_geom_vertex_array = talloc(GeomVertexArray, 1);

  (new_geom_vertex_array->vertices) = vertices;
  (new_geom_vertex_array->nV) = nV;
  (new_geom_vertex_array->num_ptrs_to) = 0;

  return new_geom_vertex_array;
}


/*--------------------------------------------------------------------------
 * GeomFreeVertexArray
 *--------------------------------------------------------------------------*/

void              GeomFreeVertexArray(
                                      GeomVertexArray *vertex_array)
{
  int v;


  (vertex_array->num_ptrs_to)--;

  /* only free up the structure when there are no pointers to it */
  if ((vertex_array->num_ptrs_to) <= 0)
  {
    for (v = 0; v < (vertex_array->nV); v++)
      tfree((vertex_array->vertices)[v]);
    tfree(vertex_array->vertices);
    tfree(vertex_array);
  }
}


/*--------------------------------------------------------------------------
 * GeomNewTIN
 *--------------------------------------------------------------------------*/

GeomTIN          *GeomNewTIN(
                             GeomVertexArray *vertex_array,
                             GeomTriangle **  triangles,
                             int              nT)
{
  GeomTIN   *new_geom_tin;


  new_geom_tin = ctalloc(GeomTIN, 1);

  (new_geom_tin->vertex_array) = vertex_array;
  (new_geom_tin->vertex_array->num_ptrs_to)++;
  (new_geom_tin->triangles) = triangles;
  (new_geom_tin->nT) = nT;

  return new_geom_tin;
}


/*--------------------------------------------------------------------------
 * GeomFreeTIN
 *--------------------------------------------------------------------------*/

void      GeomFreeTIN(GeomTIN *surface)
{
  int t;


  GeomFreeVertexArray(surface->vertex_array);
  for (t = 0; t < GeomTINNumTriangles(surface); t++)
    tfree(GeomTINTriangle(surface, t));
  tfree(surface->triangles);
  tfree(surface);
}


/*--------------------------------------------------------------------------
 * GeomNewSolid
 *--------------------------------------------------------------------------*/

GeomSolid  *GeomNewSolid(
                         void *data,
                         int   type)
{
  GeomSolid   *new_geom_solid;


  new_geom_solid = talloc(GeomSolid, 1);

  (new_geom_solid->data) = data;
  (new_geom_solid->type) = type;
  (new_geom_solid->patches) = NULL;

  return new_geom_solid;
}


/*--------------------------------------------------------------------------
 * GeomFreeSolid
 *--------------------------------------------------------------------------*/

void        GeomFreeSolid(
                          GeomSolid *solid)
{
  switch (solid->type)
  {
    case GeomTSolidType:
      GeomFreeTSolid((GeomTSolid*)(solid->data));
      break;
  }

  if (solid->patches)
    NA_FreeNameArray(solid->patches);

  tfree(solid);
}


/*--------------------------------------------------------------------------
 * GeomReadSolids
 *--------------------------------------------------------------------------*/

int           GeomReadSolids(
                             GeomSolid ***solids_ptr,
                             char *       geom_input_name,
                             int          type)
{
  GeomSolid  **solids;

  void       **solids_data = NULL;

  int i, nsolids = 0;


  switch (type)
  {
    case GeomTSolidType:
      nsolids = GeomReadTSolids((GeomTSolid***)&solids_data, geom_input_name);
      break;
  }

  solids = ctalloc(GeomSolid *, nsolids);
  for (i = 0; i < nsolids; i++)
    solids[i] = GeomNewSolid(solids_data[i], type);

  tfree(solids_data);

  *solids_ptr = solids;

  return nsolids;
}


/*--------------------------------------------------------------------------
 * GeomSolidFromBox
 *--------------------------------------------------------------------------*/

GeomSolid  *GeomSolidFromBox(
                             double xl,
                             double yl,
                             double zl,
                             double xu,
                             double yu,
                             double zu,
                             int    type)
{
  GeomSolid   *solid;

  void        *solid_data = NULL;


  switch (type)
  {
    case GeomTSolidType:
      solid_data = (void*)GeomTSolidFromBox(xl, yl, zl, xu, yu, zu);
      break;
  }

  solid = GeomNewSolid(solid_data, type);

  return solid;
}


/*--------------------------------------------------------------------------
 * IntersectLineWithTriangle
 *
 * SPECIAL NOTE :
 * This routine computes normals based on the assumption of a counter-
 * clockwise ordering of vertices for the triangle.  That is, the following
 * triangle has outward normal:
 *
 *                                  2
 *                                 /\
 *                                /  \
 *                               /    \
 *                              /      \
 *                             /        \
 *                            /__________\
 *                           0            1
 *
 *--------------------------------------------------------------------------*/

void IntersectLineWithTriangle(
                               unsigned int line_direction,
                               double       coord_0,
                               double       coord_1,
                               double       v0_x,
                               double       v0_y,
                               double       v0_z,
                               double       v1_x,
                               double       v1_y,
                               double       v1_z,
                               double       v2_x,
                               double       v2_y,
                               double       v2_z,
                               int *        intersects,
                               double *     point,
                               int *        normal_component)
{
  double p[3];
  double q[3] = { 0, 0, 0 };
  double A, B, C, D;
  double u_0, v_0, w_0;
  double u_1, v_1, w_1;
  double p0_x, p0_y, p0_z;
  double p1_x, p1_y, p1_z;
  double p0, p1, q0, q1;                                     /* real new, CHB */
  double dx, dy, dz;
  double a = 0.0, b = 0.0, coord = 0.0, differential = 0.0;
  int sign_holder, next_sign_holder, prev_sign_holder;
  int n_crossings;
  int edge_inter, vertex_inter;
  int edge_number = 0, vertex_number = 0;
  int k, kp1;

  /*---------------------------------------------------
   * Search the triangles to find intersections.
   *
   * Use the algorithm on p.56 of:
   *   Glassner, "An Introduction to Ray Tracing"
   * (see RDF for relevant pages).
   *
   * Note: we project the triangle onto the plane
   * specified by `line_direction' instead of the
   * "dominant" coordinate's plane, as in the standard
   * algorithm.
   *
   * Note: the routine is slightly modified in order
   * to consistently find edge and vertex intersections.
   * Lines in the original algorithm that were modified,
   * and new lines added to the algorithm are marked
   * with comments below.  The main modifications can
   * be summed up as follows: (1) we treat p=0 and q=0
   * explicitly (when this occurs `sign_holder' and
   * `next_sign_holder' are 0 below); (2) a `break'
   * statement below always indicates an intersection
   * on an edge or vertex.
   *---------------------------------------------------*/

  /*---------------------------------------------
   * Orient along specified direction and translate the
   * triangle so that (coord_0,coord_1) is the origin.
   *---------------------------------------------*/

  if (line_direction == XDIRECTION)
  {
    p[0] = v0_y - coord_0;
    q[0] = v0_z - coord_1;
    p[1] = v1_y - coord_0;
    q[1] = v1_z - coord_1;
    p[2] = v2_y - coord_0;
    q[2] = v2_z - coord_1;
  }
  else if (line_direction == YDIRECTION)
  {
    p[0] = v0_x - coord_0;
    q[0] = v0_z - coord_1;
    p[1] = v1_x - coord_0;
    q[1] = v1_z - coord_1;
    p[2] = v2_x - coord_0;
    q[2] = v2_z - coord_1;
  }
  else if (line_direction == ZDIRECTION)
  {
    p[0] = v0_x - coord_0;
    q[0] = v0_y - coord_1;
    p[1] = v1_x - coord_0;
    q[1] = v1_y - coord_1;
    p[2] = v2_x - coord_0;
    q[2] = v2_y - coord_1;
  }

  /*---------------------------------------------
   * Check to see how many edges cross the
   * positive u axis.  Also check to see if
   * the origin is on an edge or vertex.
   *---------------------------------------------*/

  n_crossings = 0;
  edge_inter = FALSE;
  vertex_inter = FALSE;

  *intersects = FALSE;
  *normal_component = 0;
  *point = 0.0;

  /* set previous sign holder */
  if (q[2] < 0.0)                                                /* new, CHB */
    prev_sign_holder = -1;                                       /* new, CHB */
  else if (q[2] > 0.0)                                           /* new, CHB */
    prev_sign_holder = 1;                                        /* new, CHB */
  else                                                           /* new, CHB */
    prev_sign_holder = 0;                                        /* new, CHB */

  /* set sign holder */
  if (q[0] < 0.0)
    sign_holder = -1;
  else if (q[0] > 0.0)
    sign_holder = 1;
  else                                                           /* new */
    sign_holder = 0;                                             /* new */

  for (k = 0; k < 3; k++)
  {
    kp1 = (k + 1) % 3;

    /* set next sign holder */
    if (q[kp1] < 0.0)
      next_sign_holder = -1;
    else if (q[kp1] > 0.0)
      next_sign_holder = 1;
    else                                                         /* new */
      next_sign_holder = 0;                                      /* new */

    if (sign_holder * next_sign_holder < 0)                      /* modified */
    {
      if ((p[k] > 0.0) && (p[kp1] > 0.0))
      {
        n_crossings++;
      }
      else if ((p[k] >= 0.0) || (p[kp1] >= 0.0))                 /* modified */
      {
/*          D = p[k] - q[k] * ((p[kp1] - p[k]) / (q[kp1] - q[k]));  */
        if (p[k] < p[kp1])
        {
          p0 = p[k];
          p1 = p[kp1];
          q0 = q[k];
          q1 = q[kp1];
        }
        else if (p[k] > p[kp1])
        {
          p0 = p[kp1];
          p1 = p[k];
          q0 = q[kp1];
          q1 = q[k];
        }
        else if (q[k] < q[kp1])
        {
          p0 = p[k];
          p1 = p[kp1];
          q0 = q[k];
          q1 = q[kp1];
        }
        else if (q[k] > q[kp1])
        {
          p0 = p[kp1];
          p1 = p[k];
          q0 = q[kp1];
          q1 = q[k];
        }
        else
        {
          p0 = p[k];
          p1 = p[kp1];
          q0 = q[k];
          q1 = q[kp1];
        }
        D = p0 - q0 * ((p1 - p0) / (q1 - q0));               /* real new, CHB */
        if (D > 0.0)
          n_crossings++;
        else if (D == 0.0)                                       /* new */
        {                                                        /* new, CHB */
          /* Edge intersection */
          edge_inter = TRUE;                                     /* new, CHB */
          edge_number = k;                                       /* new, CHB */
          break;                                                 /* new */
        }                                                        /* new, CHB */
      }
    }
    else if (sign_holder == 0)                                   /* new */
    {                                                            /* new */
      if (next_sign_holder == 0)                                 /* new */
      {                                                          /* new */
        if ((p[k] * p[kp1]) <= 0.0)                              /* new */
        {                                                        /* new, CHB */
          if (p[k] == 0.0)                                       /* new, CHB */
          {                                                      /* new, CHB */
            /* Vertex intersection */
            vertex_inter = TRUE;                                 /* new, CHB */
            vertex_number = k;                                   /* new, CHB */
          }                                                      /* new, CHB */
          else if (p[kp1] == 0.0)                                /* new, CHB */
          {                                                      /* new, CHB */
            /* Vertex intersection */
            vertex_inter = TRUE;                                 /* new, CHB */
            vertex_number = kp1;                                 /* new, CHB */
          }                                                      /* new, CHB */
          else                                                   /* new, CHB */
          {                                                      /* new, CHB */
            /* Edge intersection */
            edge_inter = TRUE;                                   /* new, CHB */
            edge_number = k;                                     /* new, CHB */
          }                                                      /* new, CHB */
          break;                                                 /* new */
        }                                                        /* new, CHB */
      }                                                          /* new */
      else                                                       /* new */
      {                                                          /* new */
        if (p[k] > 0.0)                                          /* new */
        {                                                        /* new, CHB */
          if (prev_sign_holder * next_sign_holder < 0)           /* new, CHB */
          {                                                      /* new, CHB */
            n_crossings++;                                       /* new, CHB */
          }                                                      /* new, CHB */
        }                                                        /* new, CHB */
        else if (p[k] == 0.0)                                    /* new, CHB */
        {                                                        /* new, CHB */
          /* Vertex intersection */
          vertex_inter = TRUE;                                   /* new, CHB */
          vertex_number = k;                                     /* new, CHB */
          break;                                                 /* new, CHB */
        }                                                        /* new, CHB */

#if 0
        if (p[k] > 0.0)                                          /* new */
          n_crossings++;                                         /* new */
        else if (p[k] == 0.0)                                    /* new */
          break;                                                 /* new */
#endif
      }                                                          /* new */
    }                                                            /* new */

    prev_sign_holder = sign_holder;                              /* new, CHB */
    sign_holder = next_sign_holder;
  }

  /*---------------------------------------------
   * If an odd number of edges cross the +u axis
   * or if we broke out of the above edge loop,
   * then we have found a triangle intersection.
   * Note that if we broke out of the above loop,
   * this indicates that the intersection is on
   * a triangle edge (or vertex).
   *---------------------------------------------*/

  if (vertex_inter || edge_inter || ((n_crossings % 2) || (k < 3)))
  {
    /* Compute the normal that will be returned, regardless of inter type */
    u_0 = v1_x - v0_x;
    v_0 = v1_y - v0_y;
    w_0 = v1_z - v0_z;
    u_1 = v2_x - v0_x;
    v_1 = v2_y - v0_y;
    w_1 = v2_z - v0_z;

    A = v_0 * w_1 - v_1 * w_0;
    B = w_0 * u_1 - w_1 * u_0;
    C = u_0 * v_1 - u_1 * v_0;
    D = A * v0_x + B * v0_y + C * v0_z;

/*
 *    if (abs(A) < .00000001) A = 0.0;
 */

    if (line_direction == XDIRECTION)
    {
      if (A != 0.0)
      {
        /* Found an intersection */
        *intersects = TRUE;

        if (A > 0.0)
        {
          *normal_component = 1;
        }
        else
        {
          *normal_component = -1;
        }
      }
    }
    if (line_direction == YDIRECTION)
    {
      if (B != 0.0)
      {
        /* Found an intersection */
        *intersects = TRUE;

        if (B > 0.0)
        {
          *normal_component = 1;
        }
        else
        {
          *normal_component = -1;
        }
      }
    }
    if (line_direction == ZDIRECTION)
    {
      if (C != 0.0)
      {
        /* Found an intersection */
        *intersects = TRUE;

        if (C > 0.0)
        {
          *normal_component = 1;
        }
        else
        {
          *normal_component = -1;
        }
      }
    }

    if (*intersects && vertex_inter)
    {
      /* Use the original vertex points to set the intersection value */
      /* Note : Vertex intersections always occur at k */
      if (line_direction == XDIRECTION)
      {
        if (vertex_number == 0)
        {
          *point = v0_x;
        }
        else if (vertex_number == 1)
        {
          *point = v1_x;
        }
        else if (vertex_number == 2)
        {
          *point = v2_x;
        }
        else      /* Shouldn't get here, but if it does - punt */
        {
          *intersects = FALSE;
        }
      }
      if (line_direction == YDIRECTION)
      {
        if (vertex_number == 0)
        {
          *point = v0_y;
        }
        else if (vertex_number == 1)
        {
          *point = v1_y;
        }
        else if (vertex_number == 2)
        {
          *point = v2_y;
        }
        else      /* Shouldn't get here, but if it does - punt */
        {
          *intersects = FALSE;
        }
      }
      if (line_direction == ZDIRECTION)
      {
        if (vertex_number == 0)
        {
          *point = v0_z;
        }
        else if (vertex_number == 1)
        {
          *point = v1_z;
        }
        else if (vertex_number == 2)
        {
          *point = v2_z;
        }
        else      /* Shouldn't get here, but if it does - punt */
        {
          *intersects = FALSE;
        }
      }
    }
    else if (*intersects && edge_inter)
    {
      /* Note : Edge intersections always occur between k and kp1 */
      if (edge_number == 0)
      {
        p0_x = v0_x;
        p0_y = v0_y;
        p0_z = v0_z;
        p1_x = v1_x;
        p1_y = v1_y;
        p1_z = v1_z;
      }
      else if (edge_number == 1)
      {
        p0_x = v1_x;
        p0_y = v1_y;
        p0_z = v1_z;
        p1_x = v2_x;
        p1_y = v2_y;
        p1_z = v2_z;
      }
      else if (edge_number == 2)
      {
        p0_x = v2_x;
        p0_y = v2_y;
        p0_z = v2_z;
        p1_x = v0_x;
        p1_y = v0_y;
        p1_z = v0_z;
      }
      else    /* Shouldn't get here, but if it does - punt */
      {
        p0_x = 0.0;
        p0_y = 0.0;
        p0_z = 0.0;
        p1_x = 0.0;
        p1_y = 0.0;
        p1_z = 0.0;
        *intersects = FALSE;
      }

#if 1
      dx = p1_x - p0_x;
      dy = p1_y - p0_y;
      dz = p1_z - p0_z;

      if (line_direction == XDIRECTION)
      {
        if (p0_y < p1_y)
        {
          a = p0_x;
          b = p0_y;
          coord = coord_0;
          differential = dx / dy;
        }
        else if (p0_y > p1_y)
        {
          a = p1_x;
          b = p1_y;
          coord = coord_0;
          differential = dx / dy;
        }
        else if (p0_z < p1_z)
        {
          a = p0_x;
          b = p0_z;
          coord = coord_1;
          differential = dx / dz;
        }
        else if (p0_z > p1_z)
        {
          a = p1_x;
          b = p1_z;
          coord = coord_1;
          differential = dx / dz;
        }
        else
        {
          /* Shouldn't happen, if it does then the two points line in  */
          /*   the same projection plane, or the loop broke out screwy */
          a = 0.0;
          b = 0.0;
          coord = 0.0;
          differential = 0.0;

          *intersects = FALSE;
        }
      }

      if (line_direction == YDIRECTION)
      {
        if (p0_x < p1_x)
        {
          a = p0_y;
          b = p0_x;
          coord = coord_0;
          differential = dy / dx;
        }
        else if (p0_x > p1_x)
        {
          a = p1_y;
          b = p1_x;
          coord = coord_0;
          differential = dy / dx;
        }
        else if (p0_z < p1_z)
        {
          a = p0_y;
          b = p0_z;
          coord = coord_1;
          differential = dy / dz;
        }
        else if (p0_z > p1_z)
        {
          a = p1_y;
          b = p1_z;
          coord = coord_1;
          differential = dy / dz;
        }
        else
        {
          /* Shouldn't happen, if it does then the two points line in  */
          /*   the same projection plane, or the loop broke out screwy */
          a = 0.0;
          b = 0.0;
          coord = 0.0;
          differential = 0.0;

          *intersects = FALSE;
        }
      }

      if (line_direction == ZDIRECTION)
      {
        if (p0_x < p1_x)
        {
          b = p0_x;
          a = p0_z;
          coord = coord_0;
          differential = dz / dx;
        }
        else if (p0_x > p1_x)
        {
          b = p1_x;
          a = p1_z;
          coord = coord_0;
          differential = dz / dx;
        }
        else if (p0_y < p1_y)
        {
          b = p0_y;
          a = p0_z;
          coord = coord_1;
          differential = dz / dy;
        }
        else if (p0_y > p1_y)
        {
          b = p1_y;
          a = p1_z;
          coord = coord_1;
          differential = dz / dy;
        }
        else
        {
          /* Shouldn't happen, if it does then the two points line in  */
          /*   the same projection plane, or the loop broke out screwy */
          a = 0.0;
          b = 0.0;
          coord = 0.0;
          differential = 0.0;

          *intersects = FALSE;
        }
      }

      *point = a + (coord - b) * differential;
#endif
#if 0
      if (line_direction == XDIRECTION)
      {
        if (p0_y < p1_y)
        {
          *point = p0_x + ((coord_0 - p0_y) / (p1_y - p0_y)) * (p1_x - p0_x);
        }
        else if (p0_y > p1_y)
        {
          *point = p1_x + ((coord_0 - p1_y) / (p0_y - p1_y)) * (p0_x - p1_x);
        }
        else if (p0_z < p1_z)
        {
          *point = p0_x + ((coord_1 - p0_z) / (p1_z - p0_z)) * (p1_x - p0_x);
        }
        else if (p0_z > p1_z)
        {
          *point = p1_x + ((coord_1 - p1_z) / (p0_z - p1_z)) * (p0_x - p1_x);
        }
        else
        {
          /* Shouldn't happen, if it does then the two points line in  */
          /*   the same projection plane, or the loop broke out screwy */
          *intersects = FALSE;
        }
      }

      if (line_direction == YDIRECTION)
      {
        if (p0_x < p1_x)
        {
          *point = p0_y + ((coord_0 - p0_x) / (p1_x - p0_x)) * (p1_y - p0_y);
        }
        else if (p0_x > p1_x)
        {
          *point = p1_y + ((coord_0 - p1_x) / (p0_x - p1_x)) * (p0_y - p1_y);
        }
        else if (p0_z < p1_z)
        {
          *point = p0_y + ((coord_1 - p0_z) / (p1_z - p0_z)) * (p1_y - p0_y);
        }
        else if (p0_z > p1_z)
        {
          *point = p1_y + ((coord_1 - p1_z) / (p0_z - p1_z)) * (p0_y - p1_y);
        }
        else
        {
          /* Shouldn't happen, if it does then the two points line in  */
          /*   the same projection plane, or the loop broke out screwy */
          *intersects = FALSE;
        }
      }

      if (line_direction == ZDIRECTION)
      {
        if (p0_x < p1_x)
        {
          *point = p0_z + ((coord_0 - p0_x) / (p1_x - p0_x)) * (p1_z - p0_z);
        }
        else if (p0_x > p1_x)
        {
          *point = p1_z + ((coord_0 - p1_x) / (p0_x - p1_x)) * (p0_z - p1_z);
        }
        else if (p0_y < p1_y)
        {
          *point = p0_z + ((coord_1 - p0_y) / (p1_y - p0_y)) * (p1_z - p0_z);
        }
        else if (p0_y > p1_y)
        {
          *point = p1_z + ((coord_1 - p1_y) / (p0_y - p1_y)) * (p0_z - p1_z);
        }
        else
        {
          /* Shouldn't happen, if it does then the two points line in  */
          /*   the same projection plane, or the loop broke out screwy */
          *intersects = FALSE;
        }
      }
#endif
    }
    else if (*intersects && ((n_crossings % 2) || (k < 3)))
    {
      /*------------------------------------------
       * If the triangle lies in a plane parallel to
       * the chosen direction.  We will assume that this
       * case is *not* an intersection since these
       * triangles are on the sides of the solid.
       * Since we are dealing with a *solid*, there
       * must exist a non-vertical triangle with
       * one edge in this same vertical plane.
       *------------------------------------------*/

      if (line_direction == XDIRECTION)
      {
        *point = (D - B * coord_0 - C * coord_1) / A;
      }
      if (line_direction == YDIRECTION)
      {
        *point = (D - A * coord_0 - C * coord_1) / B;
      }
      if (line_direction == ZDIRECTION)
      {
        *point = (D - A * coord_0 - B * coord_1) / C;
      }
    }
  }
}
