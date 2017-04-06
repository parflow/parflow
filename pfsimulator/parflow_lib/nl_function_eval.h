/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

typedef struct
{
   PFModule    *nl_function_eval;
   PFModule    *richards_jacobian_eval;
   PFModule    *precond;
   PFModule    *bc_pressure;//dok

#ifdef withTemperature   
   PFModule    *temperature_function_eval;
   PFModule    *temperature_jacobian_eval;
   PFModule    *precond_temperature;   
#endif

   ProblemData *problem_data;

   Matrix      *jacobian_matrix;
   Matrix      *jacobian_matrix_C;//dok
   Matrix      *jacobian_matrix_E;//dok
   Matrix      *jacobian_matrix_F;//dok

   N_Vector     stateContainerNVector;	

   double       dt;
   double       time;
   double       *outflow; /*sk*/
} State;



/*--------------------------------------------------------------------------
 * Accessor macros: State
 *--------------------------------------------------------------------------*/

#define StateFunc(state)          ((state)->nl_function_eval)
#define StateBCPressure(state)          ((state)->bc_pressure)//dok
#define StateProblemData(state)   ((state)->problem_data)
#define StateDt(state)            ((state)->dt)
#define StateTime(state)          ((state)->time)
#define StateJacEval(state)       ((state)->richards_jacobian_eval)
#define StateJac(state)           ((state)->jacobian_matrix)
#define StateJacC(state)           ((state)->jacobian_matrix_C)//dok
#define StateJacE(state)           ((state)->jacobian_matrix_E)//dok
#define StateJacF(state)           ((state)->jacobian_matrix_F)//dok
#define StatePrecond(state)       ((state)->precond)
#define StateOutflow(state)       ((state)->outflow) /*sk*/

/* Vectors that are part of fieldContainerNVector and may vary with number of species/phases */
#define StateFieldContainer(state)       ((state)->stateContainerNVector)

#define StateOldDensity(state)    (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[1])
#define StateOldPressure(state)   (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[4])
#define StateOldSaturation(state) (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[3])
#define StateDensity(state)       (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[0])
#define StateSaturation(state)    (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[2])
#define StateEvapTrans(state)     (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[5])
#define StateOvrlBcFlx(state)     (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[6])
#define StateXvel(state)          (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[7]) //jjb
#define StateYvel(state)          (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[8]) //jjb
#define StateZvel(state)          (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[9]) //jjb

#ifdef withTemperature
  #define StatePrecondTemperature(state)       ((state)->precond_temperature)
  #define StateFuncTemperature(state)           ((state)->temperature_function_eval)
  #define StateJacEvalTemperature(state) ((state)->temperature_jacobian_eval)

  #define StateOldTemperature(state)     (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[14])
  #define StateOldViscosity(state)       (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[13]) //does not change yet
  #define StateViscosity(state)          (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[12])
  #define StateClmEnergySource(state)    (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[15])
  #define StateForcT(state)              (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[16])
  #define StateHeatCapacityWater(state)  (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[10])
  #define StateHeatCapacityRock(state)   (((N_VectorContent)(((state)->stateContainerNVector)->content))->dims[11])
#endif
