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

typedef struct {
  PFModule    *nl_function_eval;
  PFModule    *richards_jacobian_eval;
  PFModule    *precond;
  PFModule    *bc_pressure; //dok

  ProblemData *problem_data;

  Matrix      *jacobian_matrix;
  Matrix      *jacobian_matrix_C; //dok
  Matrix      *jacobian_matrix_E; //dok
  Matrix      *jacobian_matrix_F; //dok


  Vector      *old_density;
  Vector      *old_saturation;
  Vector      *old_pressure;
  Vector      *density;
  Vector      *saturation;

  double dt;
  double time;
  double       *outflow;  /*sk*/

  Vector       *evap_trans;  /*sk*/
  Vector       *ovrl_bc_flx;  /*sk*/

  Vector       *x_velocity;  //jjb
  Vector       *y_velocity;  //jjb
  Vector       *z_velocity;  //jjb
} State;



/*--------------------------------------------------------------------------
 * Accessor macros: State
 *--------------------------------------------------------------------------*/

#define StateFunc(state)          ((state)->nl_function_eval)
#define StateBCPressure(state)          ((state)->bc_pressure)//dok
#define StateProblemData(state)   ((state)->problem_data)
#define StateOldDensity(state)    ((state)->old_density)
#define StateOldPressure(state)   ((state)->old_pressure)
#define StateOldSaturation(state) ((state)->old_saturation)
#define StateDensity(state)       ((state)->density)
#define StateSaturation(state)    ((state)->saturation)
#define StateDt(state)            ((state)->dt)
#define StateTime(state)          ((state)->time)
#define StateJacEval(state)       ((state)->richards_jacobian_eval)
#define StateJac(state)           ((state)->jacobian_matrix)
#define StateJacC(state)           ((state)->jacobian_matrix_C)//dok
#define StateJacE(state)           ((state)->jacobian_matrix_E)//dok
#define StateJacF(state)           ((state)->jacobian_matrix_F)//dok
#define StatePrecond(state)       ((state)->precond)
#define StateOutflow(state)       ((state)->outflow) /*sk*/
#define StateEvapTrans(state)     ((state)->evap_trans) /*sk*/
#define StateOvrlBcFlx(state)     ((state)->ovrl_bc_flx) /*sk*/
#define StateXvel(state)          ((state)->x_velocity) //jjb
#define StateYvel(state)          ((state)->y_velocity) //jjb
#define StateZvel(state)          ((state)->z_velocity) //jjb
