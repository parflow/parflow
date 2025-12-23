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

int main(int argc, char** argv)
{
  int i, j;
  FILE* inFile;
  FILE* outFile;
  char* inFilename = argv[1];
  char* outFilename = argv[2];
  int version;
  int numVertices;
  int numSolids;
  int numTriangles;
  int numPatches;

  inFile = fopen(inFilename, "r");
  outFile = fopen(outFilename, "w");

  fprintf(outFile, "# vtk DataFile Version 2.0\n");
  fprintf(outFile, "%s\n", inFilename);
  fprintf(outFile, "ASCII\n");

  fscanf(inFile, "%d", &version);

  fscanf(inFile, "%d", &numVertices);

  fprintf(outFile, "DATASET POLYDATA\n");
  fprintf(outFile, "POINTS %d float\n", numVertices);

  for (i = 0; i < numVertices; ++i)
  {
    double x, y, z;
    fscanf(inFile, "%lf %lf %lf", &x, &y, &z);
    fprintf(outFile, "%f %f %f\n", x, y, z);
  }

  fscanf(inFile, "%d", &numSolids);

  for (i = 0; i < numSolids; ++i)
  {
    fscanf(inFile, "%d", &numTriangles);
    fprintf(outFile, "POLYGONS %d %d\n", numTriangles, (3 + 1) * numTriangles);

    for (j = 0; j < numTriangles; ++j)
    {
      int p1, p2, p3;
      fscanf(inFile, "%d %d %d", &p1, &p2, &p3);
      fprintf(outFile, "%d %d %d %d\n", 3, p1, p2, p3);
    }
  }

  fscanf(inFile, "%d", &numPatches);
  printf("NumPatches = %d\n", numPatches);

  for (i = 0; i < numPatches; ++i)
  {
    fscanf(inFile, "%d", &numTriangles);
    printf("\tNumTriangles = %d\n", numTriangles);
    for (j = 0; j < numTriangles; ++j)
    {
      int t;
      fscanf(inFile, "%d", &t);
    }
  }

  fclose(inFile);
}
