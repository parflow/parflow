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

int main(int argc, char *argv[])
{
  /* These are input */
  int NX, NY, NZ;
  int nx, ny, nz;
  int ncnt;
  int time_step_max;


  char *input_filename;
  char *output_filename;
  /* end of input */

  char filename[1024];

  FILE *infile;

  FILE **fp;

  int c;
  int l;
  int time_step;

  double X = 0.0, Y = 0.0, Z = 0.0;
  double DX = 0.0, DY = 0.0, DZ = 0.0;

  int NumSubgrids = 1;

  int x = 0, y = 0, z = 0;
  int rx = 0, ry = 0, rz = 0;

  int i, j, k;

  int nindex;

  int *index;
  float *cnt;

  double d_value;

  if (argc != 8)
  {
    printf("Usage: slimtopfsb nx1 ny1 nz1 num_components num_dumps input_file output_file_root\n");
    exit(1);
  }

  nx = NX = atoi(argv[1]);
  ny = NY = atoi(argv[2]);
  nz = NZ = atoi(argv[3]);

  ncnt = atoi(argv[4]);

  time_step_max = atoi(argv[5]);

  input_filename = argv[6];

  output_filename = argv[7];

  /* Open up the input file (in the f77 env) */
  /* All the rest of the input is read in f77 env */
  ropen_(input_filename, (long)strlen(input_filename));

  /* Each component has a separate output file */
  fp = (FILE**)malloc(ncnt * sizeof(FILE *));

  /* For each timestep write out each component */
  for (time_step = 0; time_step < time_step_max; time_step++)
  {
    for (c = 0; c < ncnt; c++)
    {
      /* Open each of the output files */
      sprintf(filename, "%s.0.%02d.%05d.pfsb", output_filename, c, time_step);
      fp[c] = fopen(filename, "wb");

      /* Write out the header */
      fwrite(&X, sizeof(double), 1, fp[c]);
      fwrite(&Y, sizeof(double), 1, fp[c]);
      fwrite(&Z, sizeof(double), 1, fp[c]);

      fwrite(&NX, sizeof(int), 1, fp[c]);
      fwrite(&NY, sizeof(int), 1, fp[c]);
      fwrite(&NZ, sizeof(int), 1, fp[c]);

      fwrite(&DX, sizeof(double), 1, fp[c]);
      fwrite(&DY, sizeof(double), 1, fp[c]);
      fwrite(&DZ, sizeof(double), 1, fp[c]);

      fwrite(&NumSubgrids, sizeof(int), 1, fp[c]);

      fwrite(&x, sizeof(int), 1, fp[c]);
      fwrite(&y, sizeof(int), 1, fp[c]);
      fwrite(&z, sizeof(int), 1, fp[c]);

      fwrite(&nx, sizeof(int), 1, fp[c]);
      fwrite(&ny, sizeof(int), 1, fp[c]);
      fwrite(&nz, sizeof(int), 1, fp[c]);

      fwrite(&rx, sizeof(int), 1, fp[c]);
      fwrite(&ry, sizeof(int), 1, fp[c]);
      fwrite(&rz, sizeof(int), 1, fp[c]);
    }

    /* Read the number of cells for this time step */
    rnindex_(&nindex);

    /* Each component has the same number of cells */
    for (c = 0; c < ncnt; c++)
      fwrite(&nindex, sizeof(int), 1, fp[c]);

    /* Get the array to hold the indexes for each cell */
    index = (int*)malloc(nindex * sizeof(int));
    rindex_(index, &nindex);

    /* Array to hold the cell values */
    cnt = (float*)malloc(nindex * sizeof(float));

    /* For each component write out the sparse data */
    for (c = 0; c < ncnt; c++)
    {
      /* Get the values for the cells */
      rcnt_(cnt, &nindex);

      /* For each of the cells write the indices and value
       * for the sparse storage format */
      for (l = 0; l < nindex; l++)
      {
        /* calc i, j, k from the index */
        k = index[l] / (NX * NY);
        j = (index[l] - k * (NX * NY)) / NX;
        i = (index[l] - k * (NX * NY)) - j * NX;

        fwrite(&i, sizeof(int), 1, fp[c]);
        fwrite(&j, sizeof(int), 1, fp[c]);
        fwrite(&k, sizeof(int), 1, fp[c]);

        /* Need to convert f77 real to double */
        d_value = (double)cnt[l];
        fwrite(&d_value, sizeof(double), 1, fp[c]);
      }

      /* done with this component */
      fclose(fp[c]);
    }

    free(cnt);
    free(index);
  }

  free(fp);

  rclose_();
}
