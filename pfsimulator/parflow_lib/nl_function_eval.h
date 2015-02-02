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
   PFModule    *press_function_eval;
   PFModule    *richards_jacobian_eval;
   PFModule    *precond_pressure;
#ifdef withTemperature   
   PFModule    *temp_function_eval;
   PFModule    *temperature_jacobian_eval;
   PFModule    *precond_temperature;   
#endif
   PFModule    *bc_pressure;//dok

   ProblemData *problem_data;

   Matrix      *jacobian_matrix;
   Matrix      *jacobian_matrix_C;//dok
   Matrix      *jacobian_matrix_E;//dok
   Matrix      *jacobian_matrix_F;//dok


   N_Vector     fieldContainerNVector;	

   double       dt;
   double       time;
   double       *outflow; /*sk*/
   
   
} State;



/*--------------------------------------------------------------------------
 * Accessor macros: State
 *--------------------------------------------------------------------------*/

#define StatePressFunc(state)          ((state)->press_function_eval)
#define StateBCPressure(state)          ((state)->bc_pressure)//dok
#define StateProblemData(state)   ((state)->problem_data)
#define StateDt(state)            ((state)->dt)
#define StateTime(state)          ((state)->time)
#define StateJacEval(state)       ((state)->richards_jacobian_eval)
#define StateJac(state)           ((state)->jacobian_matrix)
#define StateJacC(state)           ((state)->jacobian_matrix_C)//dok
#define StateJacE(state)           ((state)->jacobian_matrix_E)//dok
#define StateJacF(state)           ((state)->jacobian_matrix_F)//dok
#define StatePrecondPressure(state)       ((state)->precond_pressure)
#ifdef withTemperature
  #define StatePrecondTemperature(state)       ((state)->precond_temperature)
  #define StateFuncTemp(state)           ((state)->temp_function_eval)
  #define StateJacEvalTemperature(state) ((state)->temperature_jacobian_eval)
#else
  #define StatePrecond(state)       ((state)->precond_pressure)
#endif
#define StateOutflow(state)       ((state)->outflow) /*sk*/
#define StateFieldContainer(state)       ((state)->fieldContainerNVector)


/* Vectors that are part of fieldContainerNVector and may vary with number of species/phases */
#define StateOldDensity(state)    (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[1])
#define StateOldPressure(state)   (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[4])
#define StateOldSaturation(state) (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[3])
#define StateDensity(state)       (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[0])
#define StateSaturation(state)    (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[2])
#define StateEvapTrans(state)     (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[5])
#define StateOvrlBcFlx(state)     (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[6])
#ifdef FGTest
    #define StateSaturation2(state)    (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[7]) /*FG for tests*/
    #define StateOldPressure2(state)   (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[9]) /*FG for tests*/
    #define StateOldSaturation2(state) (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[8]) /*FG for tests*/
#endif
#ifdef withTemperature
  #define StateOldTemperature(state)     (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[11])
  #define StateOldViscosity(state)       (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[9]) //does not change yet
  #define StateViscosity(state)          (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[9])
  #define StateClmEnergySource(state)    (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[12])
  #define StateForcT(state)              (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[13])
  #define StateHeatCapacityWater(state)  (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[7])
  #define StateHeatCapacityRock(state)   (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[8])
  #define StateXVelocity(state)   (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[14])
  #define StateYVelocity(state)   (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[15])
  #define StateZVelocity(state)   (((N_VectorContent)(((state)->fieldContainerNVector)->content))->dims[16])
#endif
