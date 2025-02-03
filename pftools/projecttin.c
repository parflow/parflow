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

#include <stdio.h>
#include <math.h>
#include "gms.h"



/*--------------------------------------------------------------------------
 * IntersectLineWithTriangle
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
                               int *        normal_component);

#define XDIRECTION 0
#define YDIRECTION 1
#define ZDIRECTION 2

/* end included stuff */


#define MAXNUM_TRIANGLES_PER_VERTEX 100
typedef struct _IntersectionList {
  int layer_id;

  double z;

  int matched;

  int global_num;

  int num_projection_triangles;
  int projection_triangles[MAXNUM_TRIANGLES_PER_VERTEX];

  struct _IntersectionList *next, *prev;
} IntersectionList;

typedef struct _MeshVertex {
  double x, y;

  int end_of_hash;

  int inspected;

  int num_triangles;
  int triangles[MAXNUM_TRIANGLES_PER_VERTEX];
  int vertex_num[MAXNUM_TRIANGLES_PER_VERTEX];  /* 0, 1, 2 */

  struct _MeshVertex *next, *prev;

  IntersectionList *ilist_head;
  IntersectionList *ilist_tail;
} MeshVertex;

typedef struct {
  int vertices[3];

  double atan2[3];
} MeshTriangle;

typedef struct {
  MeshTriangle *triangles;
  MeshVertex *vertices;

  MeshVertex *vertices_head;
  MeshVertex *vertices_tail;
  MeshVertex *current;

  int NumTriangles;
  int NumVertices;

  int LastTriangle;
  int LastVertex;
} ProjectionMesh;

double Bisector(double a0, double a1)
{
  double bisect;

  bisect = (a0 + a1) / 2;

  if ((max(a0, a1) - fmin(a0, a1)) > M_PI)
    bisect += M_PI;

  if (bisect > M_PI)
    bisect -= M_PI;

  return bisect;
}

MeshVertex *NextProjectionPoint(ProjectionMesh *PM,
                                double          max_x)
{
  if ((PM->current->next)
      && (PM->current->next->x <= max_x))
    return PM->current = PM->current->next;
  else
    return PM->current = NULL;
}

MeshVertex *FindFirstProjectionPoint(ProjectionMesh *PM, double min_x)
{
  MeshVertex *ptr;

  if ((ptr = PM->vertices_head))
  {
    while (ptr)
    {
      if (ptr->x >= min_x)
      {
        return PM->current = ptr;
      }
      ptr = ptr->next;
    }
  }

  return NULL;
}

void AddIntersectionPointToMeshVertex(MeshVertex *vertex,
                                      int         triangle,
                                      double      z_intersection,
                                      int         mat_id)
{
  IntersectionList *ptr;
  IntersectionList *new_element;

  int found;
  int duplicate;

  new_element = ctalloc(IntersectionList, 1);

  new_element->z = z_intersection;
  new_element->layer_id = mat_id;
  new_element->num_projection_triangles = 1;
  new_element->projection_triangles[0] = triangle;

  if ((ptr = vertex->ilist_head))
  {
    found = FALSE;
    duplicate = FALSE;
    while (ptr && !found && !duplicate)
    {
      if ((z_intersection == ptr->z) &&
          (mat_id == ptr->layer_id))
        duplicate = TRUE;
      else if (ptr->z > z_intersection)
        found = TRUE;
      else if ((z_intersection == ptr->z) &&
               (ptr->layer_id < mat_id))
        found = TRUE;
      else
        ptr = ptr->next;
    }

    if (duplicate)
    {
      int found_tri;
      int tri;

      for (found_tri = FALSE, tri = 0;
           !found_tri && tri < ptr->num_projection_triangles; tri++)
      {
        if (ptr->projection_triangles[tri] == triangle)
          found_tri = TRUE;
      }

      if (!found_tri)
      {
        /* Add to the existing list */
        if (ptr->num_projection_triangles > MAXNUM_TRIANGLES_PER_VERTEX)
          fprintf(stderr, "Error: Too many triangles per vertext\n");

        ptr->projection_triangles[ptr->num_projection_triangles++]
          = triangle;
      }

      free(new_element);
    }
    else if (ptr)
    {
      if (ptr->prev)
        ptr->prev->next = new_element;
      else
        vertex->ilist_head = new_element;

      new_element->next = ptr;
      new_element->prev = ptr->prev;

      ptr->prev = new_element;
    }
    else
    {
      /* new last item on list */

      vertex->ilist_tail->next = new_element;

      new_element->next = NULL;
      new_element->prev = vertex->ilist_tail;

      vertex->ilist_tail = new_element;
    }
  }
  else
  {
    /* Empty List */
    vertex->ilist_head = new_element;
    vertex->ilist_tail = new_element;

    new_element->next = NULL;
    new_element->prev = NULL;
  }
}

int AddVertexToProjectionMesh(ProjectionMesh *PM,
                              double          x,
                              double          y)
{
  int found;
  MeshVertex *ptr;

  PM->vertices[PM->LastVertex].x = x;
  PM->vertices[PM->LastVertex].y = y;

  if ((ptr = PM->vertices_head))
  {
    found = FALSE;
    while (ptr && !found)
    {
      if (ptr->x > x)
        found = TRUE;
      else
        ptr = ptr->next;
    }

    if (ptr)
    {
      if (ptr->prev)
        ptr->prev->next = &(PM->vertices[PM->LastVertex]);
      else
        PM->vertices_head = &(PM->vertices[PM->LastVertex]);

      PM->vertices[PM->LastVertex].next = ptr;
      PM->vertices[PM->LastVertex].prev = ptr->prev;

      ptr->prev = &(PM->vertices[PM->LastVertex]);
    }
    else
    {
      /* new last item on list */
      PM->vertices_tail->next = &(PM->vertices[PM->LastVertex]);

      PM->vertices[PM->LastVertex].next = NULL;
      PM->vertices[PM->LastVertex].prev = PM->vertices_tail;

      PM->vertices_tail = &(PM->vertices[PM->LastVertex]);
    }
  }
  else
  {
    /* Empty List */
    PM->vertices_head = &(PM->vertices[PM->LastVertex]);
    PM->vertices_tail = &(PM->vertices[PM->LastVertex]);

    PM->vertices[PM->LastVertex].next = NULL;
    PM->vertices[PM->LastVertex].prev = NULL;
  }

  return PM->LastVertex++;
}

void AddTriangleToProjectionMesh(ProjectionMesh *PM, gms_TIN *tin,
                                 Triangle *triangle)
{
  Vertex *v0, *v1, *v2;
  double a0, a1;

  v0 = tin->vertices[triangle->v0];
  v1 = tin->vertices[triangle->v1];
  v2 = tin->vertices[triangle->v2];

  /* Work on Vertex 0 */
  PM->triangles[PM->LastTriangle].vertices[0] = triangle->v0;
  PM->vertices[triangle->v0].triangles[
    PM->vertices[triangle->v0].num_triangles] = PM->LastTriangle;

  if (PM->vertices[triangle->v0].num_triangles > MAXNUM_TRIANGLES_PER_VERTEX)
    fprintf(stderr, "Error: maximum number of vertices exceeded\n");
  PM->vertices[triangle->v0].vertex_num[
    PM->vertices[triangle->v0].num_triangles++] = 0;

  a0 = atan2(v1->y - v0->y,
             v1->x - v0->x);

  a1 = atan2(v2->y - v0->y,
             v2->x - v0->x);

  PM->triangles[PM->LastTriangle].atan2[0] = Bisector(a0, a1);


  /* Work on Vertex 1 */
  PM->triangles[PM->LastTriangle].vertices[1] = triangle->v1;
  PM->vertices[triangle->v1].triangles[
    PM->vertices[triangle->v1].num_triangles] = PM->LastTriangle;

  if (PM->vertices[triangle->v1].num_triangles > MAXNUM_TRIANGLES_PER_VERTEX)
    fprintf(stderr, "Error: maximum number of vertices exceeded\n");
  PM->vertices[triangle->v1].vertex_num[
    PM->vertices[triangle->v1].num_triangles++] = 1;


  a0 = atan2(v0->y - v1->y,
             v0->x - v1->x);

  a1 = atan2(v2->y - v1->y,
             v2->x - v1->x);

  PM->triangles[PM->LastTriangle].atan2[1] = Bisector(a0, a1);


  /* Work on Vertex 2 */

  PM->triangles[PM->LastTriangle].vertices[2] = triangle->v2;
  PM->vertices[triangle->v2].triangles[
    PM->vertices[triangle->v2].num_triangles] = PM->LastTriangle;

  if (PM->vertices[triangle->v2].num_triangles > MAXNUM_TRIANGLES_PER_VERTEX)
    fprintf(stderr, "Error: maximum number of vertices exceeded\n");

  PM->vertices[triangle->v2].vertex_num[
    PM->vertices[triangle->v2].num_triangles++] = 2;

  a0 = atan2(v0->y - v2->y,
             v0->x - v2->x);

  a1 = atan2(v1->y - v2->y,
             v1->x - v2->x);

  PM->triangles[PM->LastTriangle].atan2[2] = Bisector(a0, a1);

  PM->LastTriangle++;
}

ProjectionMesh *NewProjectionMesh(int num_triangles, int num_vertices)
{
  ProjectionMesh *PM;

  PM = ctalloc(ProjectionMesh, 1);

  PM->NumTriangles = num_triangles;
  PM->triangles = ctalloc(MeshTriangle, num_triangles + 1);

  PM->NumVertices = num_vertices;
  PM->vertices = ctalloc(MeshVertex, num_vertices + 1);

  PM->LastVertex = 0;
  PM->LastTriangle = 0;

  return PM;
}

ProjectionMesh *ConvertTINToProjectionMesh(gms_TIN *tin)
{
  ProjectionMesh *PM;

  int index;

  PM = NewProjectionMesh(tin->ntriangles, tin->nvertices);

  for (index = 0; index < tin->nvertices; index++)
  {
    AddVertexToProjectionMesh(PM,
                              tin->vertices[index]->x,
                              tin->vertices[index]->y);
  }

  for (index = 0; index < tin->ntriangles; index++)
  {
    AddTriangleToProjectionMesh(PM, tin, tin->triangles[index]);
  }

  return PM;
}

void AddTINToProjectionMesh(ProjectionMesh *PM, gms_TIN *tin)
{
  MeshVertex *projection_point;
  int index;
  Triangle *triangle;

  Vertex *v0, *v1, *v2;

  double min_x, max_x;

  int i;

  int intersects;
  double z_intersection;
  int normal_component;




  for (index = 0; index < tin->ntriangles; index++)
  {
    triangle = tin->triangles[index];

    v0 = tin->vertices[triangle->v0];
    v1 = tin->vertices[triangle->v1];
    v2 = tin->vertices[triangle->v2];

    /* Compute the lower bound of the triangle */
    min_x = fmin(v0->x, v1->x);
    min_x = fmin(min_x, v2->x);

    /* Compute the upper bound of the triangle */
    max_x = max(v0->x, v1->x);
    max_x = max(max_x, v2->x);

    projection_point = FindFirstProjectionPoint(PM, min_x);

    intersects = FALSE;
    while (projection_point)
    {
      IntersectLineWithTriangle(ZDIRECTION,
                                projection_point->x, projection_point->y,
                                v0->x, v0->y, v0->z,
                                v1->x, v1->y, v1->z,
                                v2->x, v2->y, v2->z,
                                &intersects, &z_intersection, &normal_component);

      if (intersects)
      {
        for (i = 0; i < projection_point->num_triangles; i++)
        {
          //MeshTriangle *projection_triangle;
          //projection_triangle = &(PM->triangles[projection_point->triangles[i]]);
          /* Add Intersection point to the intersection list */
          AddIntersectionPointToMeshVertex(projection_point,
                                           projection_point->triangles[i],
                                           z_intersection,
                                           tin->mat_id);
        }
      }
      projection_point = NextProjectionPoint(PM, max_x);
    }
  }
}

void ProjectionMeshTo3DMesh(ProjectionMesh *PM, char *filename)
{
  int index;
  MeshTriangle *projection_triangle;
  int vertex_num;

  int volume_num = 1;
  int node_number = 1;

  IntersectionList *ptr;
  IntersectionList *top_v0, *top_v1, *top_v2;
  IntersectionList *bot_v0, *bot_v1, *bot_v2;

  FILE *of;

  int found_v1, found_v2;
  int tri_v0, tri_v1, tri_v2;

  int zero_volume_prism = 0;

  int num_top;

  of = fopen(filename, "w");

  fprintf(of, "MESH3D\n");

  for (index = 0; index < PM->NumTriangles; index++)
  {
    projection_triangle = &(PM->triangles[index]);

    /* Mark all intersection points not matched */
    for (vertex_num = 0; vertex_num < 3; vertex_num++)
    {
      ptr = PM->
            vertices[projection_triangle->vertices[vertex_num]].ilist_head;
      while (ptr)
      {
        ptr->matched = FALSE;
        ptr = ptr->next;
      }
    }

    num_top = 0;
    top_v0 = NULL;
    top_v1 = NULL;
    top_v2 = NULL;

    bot_v0 = PM->vertices[projection_triangle->vertices[0]].ilist_head;
    while (bot_v0)
    {
      for (tri_v0 = 0; tri_v0 < bot_v0->num_projection_triangles; tri_v0++)
      {
        if (bot_v0->projection_triangles[tri_v0] == index)
        {
          /* find matching point on v1 */
          bot_v1 = PM->vertices[projection_triangle->
                                vertices[1]].ilist_head;
          found_v1 = FALSE;
          while (bot_v1 && !found_v1)
          {
            for (tri_v1 = 0;
                 !found_v1 && tri_v1 < bot_v1->num_projection_triangles;
                 tri_v1++)
            {
              if ((!bot_v1->matched) &&
                  (bot_v1->projection_triangles[tri_v1] == index) &&
                  (bot_v1->layer_id == bot_v0->layer_id))
                found_v1 = TRUE;
            }

            if (!found_v1)
              bot_v1 = bot_v1->next;
          }


          /* if no match found don't continue to look */
          if (bot_v1)
          {
            /* find matching point on v2 */
            bot_v2 = PM->vertices[projection_triangle->
                                  vertices[2]].ilist_head;
            found_v2 = FALSE;
            while (bot_v2 && !found_v2)
            {
              for (tri_v2 = 0;
                   !found_v2 && tri_v2 < bot_v2->num_projection_triangles;
                   tri_v2++)
              {
                if ((!bot_v2->matched) &&
                    (bot_v2->projection_triangles[tri_v2] == index) &&
                    (bot_v2->layer_id == bot_v0->layer_id))
                {
                  found_v2 = TRUE;
                }
              }

              if (!found_v2)
                bot_v2 = bot_v2->next;
            }

            /* if no match found then don't add */
            if (bot_v2)
            {
              /* We matched all the top interersection points so don't
               * use in another layer */
              bot_v0->matched = TRUE;
              bot_v1->matched = TRUE;
              bot_v2->matched = TRUE;

              /* Check if this prism has zero volume....if so
               * we don't want to include it */

              if (!bot_v0->global_num)
                bot_v0->global_num = node_number++;

              if (!bot_v1->global_num)
                bot_v1->global_num = node_number++;

              if (!bot_v2->global_num)
                bot_v2->global_num = node_number++;

              /* If this is the first one the just set top and look
               * for the bottom, otherwise we need to emit a new
               * volume */
              if (top_v0 != NULL)
              {
                if (bot_v0->z == top_v0->z &&
                    bot_v1->z == top_v1->z &&
                    bot_v2->z == top_v2->z)
                {
                  zero_volume_prism++;
                }
                else
                {
                  fprintf(of, "E6W %d ", volume_num++);

                  fprintf(of, "%d %d %d ",
                          top_v0->global_num,
                          top_v1->global_num,
                          top_v2->global_num);

                  fprintf(of, "%d %d %d ",
                          bot_v0->global_num,
                          bot_v1->global_num,
                          bot_v2->global_num);

                  fprintf(of, "%d\n",
                          bot_v0->layer_id * 1000 +
                          top_v0->layer_id);
                }
              }
              /* Set the top of the next layer to the bottom of the last
               * layer */
              num_top++;
              top_v0 = bot_v0;
              top_v1 = bot_v1;
              top_v2 = bot_v2;
            }
          }
        }
      }

      bot_v0 = bot_v0->next;
    }

    if (num_top != 4)
      printf("Num_Top = %d\n", num_top);
  }

  for (index = 0; index < PM->NumVertices; index++)
  {
    ptr = PM->vertices[index].ilist_head;
    while (ptr)
    {
      if (ptr->global_num)
      {
        fprintf(of, "GN %d %.15e %.15e %.15e\n", ptr->global_num,
                PM->vertices[index].x,
                PM->vertices[index].y,
                ptr->z);
      }
      ptr = ptr->next;
    }
  }

  fclose(of);

  if (zero_volume_prism)
    fprintf(stderr, "Warning: Deleted %d zero volume prisms\n", zero_volume_prism);
}

int main(int argc, char **argv)
{
  gms_TIN     **tins;

  int num_tins;

  int i, j;

  ProjectionMesh *ProjectionMesh;

  if (argc < 3)
  {
    fprintf(stderr,
            "Usage:  projecttin <TIN with the project triangles> <TINs to project> <3D mesh output>\n");
    exit(1);
  }

  /* Read in the GMS file with the TIN that defines the project
   * triangles */


  gms_ReadTINs(&tins, &num_tins, argv[1]);

  if (num_tins > 1)
  {
    fprintf(stderr, "Can\'t use a multiple TIN file for projection\n");
    exit(1);
  }

  ProjectionMesh = ConvertTINToProjectionMesh(tins[0]);

  tfree(tins[0]->vertices);
  tfree(tins[0]->triangles);
  tfree(tins);

  for (i = 2; i < (argc - 1); i++)
  {
    /* read the TINs in next input file */
    gms_ReadTINs(&tins, &num_tins, argv[i]);

    /* Project each of the TINS using the ProjectionMesh and store
     * result into the ProjectionMesh */
    for (j = 0; j < num_tins; j++)
      AddTINToProjectionMesh(ProjectionMesh, tins[j]);

    tfree(tins[0]->vertices);
    tfree(tins[0]->triangles);
    tfree(tins);
  }

  ProjectionMeshTo3DMesh(ProjectionMesh, argv[argc - 1]);

  return 0;
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
  double p[3], q[3];
  double A, B, C, D;
  double u_0, v_0, w_0;
  double u_1, v_1, w_1;
  double p0_x, p0_y, p0_z;
  double p1_x, p1_y, p1_z;
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
  else
  {
    p[0] = NAN;
    q[0] = NAN;
    p[1] = NAN;
    q[1] = NAN;
    p[2] = NAN;
    q[2] = NAN;
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
        D = p[k] - q[k] * ((p[kp1] - p[k]) / (q[kp1] - q[k]));
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
