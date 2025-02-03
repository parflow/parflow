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
#include <stdlib.h>
#include <string.h>

#include "file_versions.h"


int FaceExists(int node, int num_vertices, int *vertices);

/* to work on
 * keep track of counts while constructing in an array to save
 * 2 pass stuff
 */

int swapped = 0;

/* the Maximum number of faces a node can belong to */

/* sgs make this a env variable with default */
#define MaxFaces 200
#define MaxMatIDs 200

#define GE6_NumFaces 5
#define GE6_NumTri 8
int GE6_FaceArray[5][5] = {
  { 3, 0, 2, 1, -1 },
  { 3, 3, 4, 5, -1 },
  { 4, 0, 1, 4, 3 },
  { 4, 1, 2, 5, 4 },
  { 4, 2, 0, 3, 5 }
};
int GE6_OppositeArray[5][2] = {
  { 3, -1 },
  { 0, -1 },
  { 2, 5 },
  { 0, 3 },
  { 1, 4 }
};


typedef struct {
  /* location in space */
  double x, y, z;

  /* list of faces that this node belong to */
  int num_faces;
  int faces[MaxFaces];
} Node;

typedef struct {
  /* nodes that define the boundaries of this face */
  int num_nodes;
  int nodes[6];

  /* the color of this face matches the matid for the element this
   * face is a member of */
  int color;

  /* the color the boundary patch for this face */
  int patch_color;

  /* index of the face which is adjacent to this one
   * -1 if there is none (ie this is a boundary face) */
  int sister_face;

  /* the index for this face in the output file */
  int output_index;
} Face;

typedef struct {
  /* the GMS type of this element */
  int gms_type;

  /* The nodes indexes for the vertices of this element */
  int num_nodes;
  int nodes[8];

  /* The faces for this element */
  int num_faces;
  int faces[8];

  /* The material id */
  int color;
} Element;

typedef struct {
  /* The faces that this patch belongs to possibly more than one
   * since faces are triangles */
  int num_faces;
  int faces[2];

  /* The boundary naming of this patch */
  int color;
} Patch;


/* Global arrays to hold all the current domain information */
Node    *nodes;
Face    *faces;
Element *elements;
Patch   *patches;

int num_nodes;
int num_faces;
int num_elements;
int num_patches;

int num_colors;
int mat_ids[MaxMatIDs];
int num_patch_colors;

void AddFace(int id, int face, int *vertices, int opposite_pnt)
{
  int j;

  double u[2];
  double v[2];
  double w[2];

  double normal[3];
  double opposite[3];

  double projection;

  int tmp;

  /* Determine if a face with the same vertices as this face exists */
  faces[face].sister_face = FaceExists(vertices[0], 3, vertices);

  if (faces[face].sister_face != -1)
  {
    if (faces[faces[face].sister_face].sister_face != -1)
      /* If the sister face already has a matching face then there are
       * more than 2 faces with same vertices which is an error */
      fprintf(stderr, "Error: Found face that has more than 1 sister face\n");
    else
      /* Make the sister point to this face.  Set up the
       * sibling relationship */
      faces[faces[face].sister_face].sister_face = face;
  }

  /* Currently a face can only be a triangle */
  faces[face].num_nodes = 3;

  /* We need to check the normals on things since sometimes we
   * elements that are out of order */

  /* compute normal */
  u[0] = nodes[vertices[1]].x - nodes[vertices[0]].x;
  v[0] = nodes[vertices[1]].y - nodes[vertices[0]].y;
  w[0] = nodes[vertices[1]].z - nodes[vertices[0]].z;

  u[1] = nodes[vertices[2]].x - nodes[vertices[0]].x;
  v[1] = nodes[vertices[2]].y - nodes[vertices[0]].y;
  w[1] = nodes[vertices[2]].z - nodes[vertices[0]].z;

  normal[0] = v[0] * w[1] - v[1] * w[0];
  normal[1] = w[0] * u[1] - w[1] * u[0];
  normal[2] = u[0] * v[1] - u[1] * v[0];

  /* get the vector pointing to the inside of the element */
  opposite[0] = nodes[opposite_pnt].x - nodes[vertices[0]].x;
  opposite[1] = nodes[opposite_pnt].y - nodes[vertices[0]].y;
  opposite[2] = nodes[opposite_pnt].z - nodes[vertices[0]].z;

  /* compute the projection of the normal on the vector pointing to the
   * inside of the element */
  projection = opposite[0] * normal[0] + opposite[1] * normal[1] +
               opposite[2] * normal[2];

  /* if normal is pointing the same direction as the inwared pointing
   * vector then we need to reorder to get outward pointing normal */
  if (projection > 0)
  {
    tmp = vertices[0];
    vertices[0] = vertices[1];
    vertices[1] = tmp;
  }

  /* Add the vertices to this face and set point the nodes back
   * to this face (so we know which faces a node belongs to */
  for (j = 0; j < 3; j++)
  {
    faces[face].nodes[j] = vertices[j];
    nodes[faces[face].nodes[j]].faces[nodes[faces[face].nodes[j]].num_faces++] = face;
    if (nodes[faces[face].nodes[j]].num_faces > MaxFaces)
      fprintf(stderr, "MaxFaces exceeded!!! %d\n",
              nodes[faces[face].nodes[j]].num_faces);
  }

  /* Set up the color of the face for outputting boundary info */
  /* color matches the color of the element */
  faces[face].color = elements[id].color;

  /* Patch color, by default make it -1 to indicate no patch */
  faces[face].patch_color = -1;
}

int FaceExists(int node, int num_vertices, int *vertices)
{
  int i, j, k;

  int found_face;
  int found_node = 0;

  /* For each face this node belongs to */
  found_face = 0;
  for (i = 0; (i < nodes[node].num_faces) && !found_face; i++)
  {
    /* Check if the number of vertices is the same for both faces
     * if not then this can't be the same face */
    if (faces[nodes[node].faces[i]].num_nodes == num_vertices)
    {
      /* Check if the vertices of the two faces are the same */
      for (j = 0; j < num_vertices; j++)
      {
        found_node = 0;
        for (k = 0; (k < num_vertices) && !found_node; k++)
        {
          if (faces[nodes[node].faces[i]].nodes[j] == vertices[k])
          {
            found_node = 1;
          }
        }

        if (!found_node)
        {
          /* Skip out of j loop, since one of the vertices was not
           * found on this face */
          break;
        }
      }

      /* If all of the vertices were found then the face was found */
      if (found_node)
        found_face = 1;
    }
  }

  if (found_face)
    return nodes[node].faces[i - 1];
  else
    return -1;
}

int main(int argc, char **argv)
{
  char line[81];
  char string[81];

  int id;
  int vertices[8];
  int face;
  int color;
  int patch;
  int i, j, f, k, c;
  double x, y, z;

  int face_count;
  int patch_count;

  int default_patch_num;

  int output_index;

  int z_same_count;

  FILE *meshfile, *pfsol, *bcfile, *mfile;

  int topbot_count;

  if (argc != 8)
  {
    fprintf(stderr, "Usage:  gmsfemtopfsol file.3dm file.3bc num_nodes num_elements num_bc output.pfsol output.mmap\n");

    exit(1);
  }

  num_nodes = atoi(argv[3]);
  num_elements = atoi(argv[4]);
  num_patches = atoi(argv[5]);

  /* allocate the arrays for the nodes etc */
  if ((nodes = (Node*)calloc(num_nodes + 1, sizeof(Node))) == NULL)
  {
    fprintf(stderr, "Error: out of memory\n");
    exit(1);
  }

  if ((elements = (Element*)calloc(num_elements + 1, sizeof(Element))) == NULL)
  {
    fprintf(stderr, "Error: out of memory\n");
    exit(1);
  }

  if ((patches = (Patch*)calloc(num_patches, sizeof(Patch))) == NULL)
  {
    fprintf(stderr, "Error: out of memory\n");
    exit(1);
  }


  num_faces = 0;
  num_colors = 0;

  if ((meshfile = fopen(argv[1], "r")) == NULL)
  {
    fprintf(stderr, "Error: can't open %s\n", argv[1]);
    exit(1);
  }

  /* Parse the input file for elements and vertices */
  while (fgets(line, 80, meshfile))
  {
    if (!strncmp(line, "GE6", 3))
    {
      /* This is a 6 vertices prism */

      sscanf(line, "%s %d %d %d %d %d %d %d %d", string,
             &id, &vertices[0], &vertices[1], &vertices[2], &vertices[3],
             &vertices[4], &vertices[5], &color);

      elements[id].gms_type = 1;

      elements[id].num_nodes = 6;
      for (i = 0; i < 6; i++)
        elements[id].nodes[i] = vertices[i];

      elements[id].num_faces = GE6_NumTri;
      num_faces += GE6_NumTri;

      elements[id].color = color;

      /* Check if this is a recorded mat_id */
      for (c = 0; c < num_colors; c++)
      {
        if (mat_ids[c] == color)
        {
          break;
        }
      }

      /* if we did not find mat_id in the list then add it */
      if (c == num_colors)
      {
        mat_ids[c] = color;
        num_colors++;
      }
    }

    /* A virtex (node) entry */
    if (!strncmp(line, "GN", 2))
    {
      sscanf(line, "%s %d %lf %lf %lf", string, &id, &x, &y, &z);

      nodes[id].x = x;
      nodes[id].y = y;
      nodes[id].z = z;

      nodes[id].num_faces = 0;
    }
  }

  fclose(meshfile);

  if ((faces = (Face*)calloc(num_faces, sizeof(Face))) == NULL)
  {
    fprintf(stderr, "Error: out of memory\n");
    exit(1);
  }

  face = 0;
  for (id = 1; id <= num_elements; id++)
  {
    if (elements[id].num_nodes == 6)
    {
      int element_face;

      /* Check for malformed prism..... output of elements of GMS
      * that are malformed due to intersecting layers
      * Figure out which point lies on top/bottom of the other */
      topbot_count = 0;
      for (j = 0; j < 3; j++)
      {
        /* Assumes top/bottom vertices are offset by 3 */
        if (nodes[elements[id].nodes[ j ]].z >
            nodes[elements[id].nodes[ j + 3 ]].z)
          topbot_count++;
        else
          topbot_count--;
      }

      /* if not all three on top/bottom then we have a problem */
      if (abs(topbot_count) != 3)
      {
        fprintf(stderr, "Error:  Malformed prism (id=%d) ParFlow will not work\n", id);
      }

      /* check if this element has no volume in z, if so then don't do
       * anything with it */
      z_same_count = 0;
      for (j = 0; j < 3; j++)
      {
        if (nodes[elements[id].nodes[j]].z ==
            nodes[elements[id].nodes[j + 3]].z)
          z_same_count++;
      }
      if (z_same_count > 2)
      {
        printf("Skipping zero volume element z %d\n", id);
        continue;
      }

      /* check if the element has zero volume in x or y dims */
      if ((nodes[elements[id].nodes[0]].x ==
           nodes[elements[id].nodes[1]].x) &&
          (nodes[elements[id].nodes[0]].y ==
           nodes[elements[id].nodes[1]].y))
      {
        printf("Skipping zero volume element z %d\n", id);
        continue;
      }

      if ((nodes[elements[id].nodes[0]].x ==
           nodes[elements[id].nodes[2]].x) &&
          (nodes[elements[id].nodes[0]].y ==
           nodes[elements[id].nodes[2]].y))
      {
        printf("Skipping zero volume element z %d\n", id);
        continue;
      }

      if ((nodes[elements[id].nodes[1]].x ==
           nodes[elements[id].nodes[2]].x) &&
          (nodes[elements[id].nodes[1]].y ==
           nodes[elements[id].nodes[2]].y))
      {
        printf("Skipping zero volume element z %d\n", id);
        continue;
      }

      /* For each side of this element add it to the face list */
      element_face = 0;
      for (f = 0; f < GE6_NumFaces; f++)
      {
        switch (GE6_FaceArray[f][0])
        {
          case 3:

            /* set up the temporary array of vertices for the
             * triangle */
            for (j = 0; j < 3; j++)
            {
              vertices[j] = elements[id].nodes[ GE6_FaceArray[f][j + 1] ];
            }

            /* Find an vertex that is on the "inside" of of the
             * triangle for computing the normal */
            for (j = 0; j < 3; j++)
            {
              if (nodes[elements[id].nodes[ GE6_FaceArray[f][j + 1] ]].z !=
                  nodes[elements[id].nodes[ (GE6_FaceArray[f][j + 1] + 3)
                                            % 6]].z)
                break;
            }

            AddFace(id, face, vertices,
                    elements[id].nodes[ (GE6_FaceArray[f][j + 1] + 3) % 6]);

            elements[id].faces[element_face++] = face++;
            break;

          case 4:
          {
            if (elements[id].nodes[ GE6_FaceArray[f][1]] <
                elements[id].nodes[ GE6_FaceArray[f][2]])
            {
              /* Add the first triangle (face) of this side */
              for (j = 0; j < 3; j++)
              {
                vertices[j] = elements[id].nodes[ GE6_FaceArray[f][j + 1] ];
              }

              AddFace(id, face, vertices,
                      elements[id].nodes[ GE6_OppositeArray[f][0]]);

              elements[id].faces[element_face++] = face++;

              /* Add the second triangle of this side */
              vertices[0] = elements[id].nodes[ GE6_FaceArray[f][3] ];
              vertices[1] = elements[id].nodes[ GE6_FaceArray[f][4] ];
              vertices[2] = elements[id].nodes[ GE6_FaceArray[f][1] ];

              AddFace(id, face, vertices,
                      elements[id].nodes[ GE6_OppositeArray[f][1]]);
              elements[id].faces[element_face++] = face++;
            }
            else
            {
              /* Add the first triangle (face) of this side */
              for (j = 0; j < 3; j++)
              {
                vertices[j] = elements[id].nodes[ GE6_FaceArray[f][j + 2] ];
              }

              AddFace(id, face, vertices,
                      elements[id].nodes[ GE6_OppositeArray[f][0]]);
              elements[id].faces[element_face++] = face++;

              /* Add the second triangle of this side */
              vertices[0] = elements[id].nodes[ GE6_FaceArray[f][4] ];
              vertices[1] = elements[id].nodes[ GE6_FaceArray[f][1] ];
              vertices[2] = elements[id].nodes[ GE6_FaceArray[f][2] ];

              AddFace(id, face, vertices,
                      elements[id].nodes[ GE6_OppositeArray[f][1]]);
              elements[id].faces[element_face++] = face++;
            }
          }
          break;
        }
      }

      elements[id].num_faces = GE6_NumTri;
    }
    else
    {
      fprintf(stderr, "Error: I don't know how to work on an element with %d nodes\n", elements[id].num_nodes);
    }

    if (swapped % 8)
      fprintf(stderr, "Error on id %d\n", id);
  }

  /* Read in the boundary patch information */
  patch = 0;

  if ((bcfile = fopen(argv[2], "r")) == NULL)
  {
    fprintf(stderr, "Error: can't open file %s\n", argv[2]);
    exit(1);
  }

  if ((mfile = fopen(argv[7], "w")) == NULL)
  {
    fprintf(stderr, "Error: can't open %s\n", argv[6]);
    exit(1);
  }


  /* patch color 0 is the default for things not colored by the
   * user */
  num_patch_colors = 1;

  fprintf(mfile, "Boundary Mapping\n\n");
  fprintf(mfile, "User value     Index\n");
  fprintf(mfile, "==========     =====\n");
  fprintf(mfile, "Default          0\n");
  while (fgets(line, 80, bcfile))
  {
    double label;

    if (!strncmp(line, "XY1", 3))
    {
      sscanf(line, "%s %d", string, &i);
      fgets(line, 80, bcfile);
      sscanf(line, "%s %lf", string, &label);

      fprintf(mfile, "%f              %d\n", label, i);
    }

    if (!strncmp(line, "CB1", 3))
    {
      /* This is a boundary specification */
      /* First value is the element index, second is a the side
       * that the boundary is on, followed by the color */
      /* CB1   867     3     1 */

      sscanf(line, "%s %d %d %d", string, &id, &face, &color);

      face--;

      if (color > num_patch_colors)
        num_patch_colors = color;

      /* Based on what side this is we have to add the boundary
       * condition to the triangles that made up that side.  The
       * vertices might have moved around so we need to search for
       * the proper face */
      switch (GE6_FaceArray[face][0])
      {
        case 3:
          for (j = 0; j < 3; j++)
            vertices[j] = elements[id].nodes[ GE6_FaceArray[face][j + 1]];

          patches[patch].faces[0] = FaceExists(vertices[0], 3, vertices);
          patches[patch].color = color;

          faces[patches[patch].faces[0]].patch_color = color;

          patches[patch].num_faces = 1;
          patch++;
          break;

        case 4:
          if (elements[id].nodes[ GE6_FaceArray[face][1]] <
              elements[id].nodes[ GE6_FaceArray[face][2]])
          {
            for (j = 0; j < 3; j++)
              vertices[j] = elements[id].nodes[ GE6_FaceArray[face][j + 1]];

            patches[patch].faces[0] = FaceExists(vertices[0], 3, vertices);
            faces[patches[patch].faces[0]].patch_color = color;

            vertices[0] = elements[id].nodes[ GE6_FaceArray[face][3] ];
            vertices[1] = elements[id].nodes[ GE6_FaceArray[face][4] ];
            vertices[2] = elements[id].nodes[ GE6_FaceArray[face][1] ];

            patches[patch].faces[1] = FaceExists(vertices[0], 3, vertices);
            faces[patches[patch].faces[1]].patch_color = color;

            patches[patch].color = color;

            patches[patch].num_faces = 2;
            patch++;
          }
          else
          {
            for (j = 0; j < 3; j++)
              vertices[j] = elements[id].nodes[ GE6_FaceArray[face][j + 2]];

            patches[patch].faces[0] = FaceExists(vertices[0], 3, vertices);
            faces[patches[patch].faces[0]].patch_color = color;

            vertices[0] = elements[id].nodes[ GE6_FaceArray[face][4] ];
            vertices[1] = elements[id].nodes[ GE6_FaceArray[face][1] ];
            vertices[2] = elements[id].nodes[ GE6_FaceArray[face][2] ];

            patches[patch].faces[1] = FaceExists(vertices[0], 3, vertices);
            faces[patches[patch].faces[1]].patch_color = color;

            patches[patch].color = color;

            patches[patch].num_faces = 2;
            patch++;
          }
          break;
      }
    }
  }

  if ((pfsol = fopen(argv[6], "w")) == NULL)
  {
    fprintf(stderr, "Error: can't open %s\n", argv[6]);
    exit(1);
  }

  fprintf(pfsol, "%d\n", PFSOL_BGMSFEM2PFSOL_VERSION);

  /* Output the node array */
  fprintf(pfsol, "%d\n", num_nodes);
  for (i = 1; i <= num_nodes; i++)
    fprintf(pfsol, "%.15e %.15e %.15e\n",
            nodes[i].x, nodes[i].y, nodes[i].z);
  fprintf(pfsol, "%d\n", num_colors + 1);

  /* Output the domain */
  face_count = 0;
  for (j = 0; j < num_faces; j++)
  {
    if (faces[j].sister_face == -1)
      face_count++;
  }
  fprintf(pfsol, "%d\n", face_count);


  default_patch_num = 0;
  output_index = 0;
  for (j = 0; j < num_faces; j++)
  {
    if (faces[j].sister_face == -1)
    {
      /* if this boundary face is not part of a patch then make it
       * part of the default patch */
      if (faces[j].patch_color == -1)
      {
        default_patch_num++;
        faces[j].patch_color = 0;
      }

      fprintf(pfsol, "%d %d %d\n",
              faces[j].nodes[0] - 1,
              faces[j].nodes[1] - 1,
              faces[j].nodes[2] - 1);
      faces[j].output_index = output_index++;
    }
  }

  /* Write out the boundary conditions on the domain
   * Each color is a different patch */
  fprintf(pfsol, "%d\n", num_patch_colors + 1);

  /* Write out the default patch where user did not specify anything */
  fprintf(pfsol, "%d\n", default_patch_num);
  if (default_patch_num)
  {
    for (j = 0; j < num_faces; j++)
    {
      if (faces[j].patch_color == 0)
      {
        fprintf(pfsol, "%d\n", faces[j].output_index);
      }
    }
  }
  else
  {
    fprintf(mfile, "\n\nNo Default patches where found\n\n");
  }

  for (c = 1; c <= num_patch_colors; c++)
  {
    patch_count = 0;
    for (j = 0; j < num_patches; j++)
    {
      if (patches[j].color == c)
      {
        patch_count += patches[j].num_faces;
      }
    }

    fprintf(pfsol, "%d\n", patch_count);

    patch_count = 0;
    for (j = 0; j < num_patches; j++)
    {
      if (patches[j].color == c)
      {
        for (k = 0; k < patches[j].num_faces; k++)
        {
          fprintf(pfsol, "%d\n", faces[patches[j].faces[k]].output_index);
        }
      }
    }
  }


  /* Output a table so user knows what index to use */
  fprintf(mfile, "\nTable Geounit MatID naming\n\n");
  fprintf(mfile, "User value     Index\n");
  fprintf(mfile, "==========     =====\n");

  /* For each of the material ids */
  for (i = 0; i < num_colors; i++)
  {
    fprintf(mfile, "%d              %d\n", mat_ids[i], i + 1);

    face_count = 0;
    for (j = 0; j < num_faces; j++)
    {
      if ((faces[j].color == mat_ids[i]) &&
          ((faces[j].sister_face == -1) ||
           (faces[j].color != faces[faces[j].sister_face].color)))
        face_count++;
    }
    fprintf(pfsol, "%d\n", face_count);

    /* output the faces that form a boundary around this
    * geounit (might be several disconnected regions) */
    output_index = 0;
    for (j = 0; j < num_faces; j++)
    {
      if ((faces[j].color == mat_ids[i]) &&
          ((faces[j].sister_face == -1) ||
           (faces[j].color != faces[faces[j].sister_face].color)))
      {
        fprintf(pfsol, "%d %d %d\n",
                faces[j].nodes[0] - 1,
                faces[j].nodes[1] - 1,
                faces[j].nodes[2] - 1);
        faces[j].output_index = output_index++;
      }
    }

    /* no patches on the geounits */
    fprintf(pfsol, "0\n");
  }

  return 0;
}
