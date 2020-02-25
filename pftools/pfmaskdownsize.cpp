/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2018, Lawrence Livermore National Security,
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

#include "simplify.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

#define vertexIndex(i, j, k) (((nx+1) * (ny+1) * (k)) + ((j) * (nx+1)) + (i))
#define triangleIndex(i, j, k) ((nx * ny * (k)) + ((j) * nx) + (i))

int main(int argc, char **argv)
{

  string inFilename(argv[1]);
  string outFilename(argv[2]);

  ifstream mask(inFilename);

  int nx, ny, nz;
  double sx = 0, sy = 0, sz = 0;
  double dx = 1000.0, dy = 1000.0 , dz = 1000.0;

  mask >> nx >> ny >> nz;

  cout << "NX = " << nx << " NY =" << ny << " NZ = " << nz << std::endl;

  assert(nz == 1);

  vector<char> indicators(nx*ny);

  for(int j = 0; j < ny; ++j)
  {
    for(int i = 0; i < nx; ++i)
    {
      int indicator;
      mask >> indicator;
      indicators[ triangleIndex(i,j,0) ] = indicator;
    }
  }

  mask.close();

  ofstream outFile(outFilename);

  int new_sx = 0;
  int new_sy = 0;
  int new_nx = 3000;
  int new_ny = 1000;
  int new_nz = 1;

  outFile << new_nx << " " << new_ny << " " << new_nz << std::endl;

  std::cout << "Start " << new_sx << "," << new_sy << std::endl;
  std::cout << "Size " << new_nx<< "," << new_ny << std::endl;
    
  for(int j = new_sy; j < new_ny + new_sy; ++j)
  {
    for(int i = new_sx; i < new_nx + new_sx; ++i)
    {
      int indicator = indicators[ triangleIndex(i,j,0) ];
      outFile << indicator << std::endl;
    }
  }

  outFile.close();

}
