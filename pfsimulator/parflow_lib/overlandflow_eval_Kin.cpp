/* PFCUDA_COMP_UNIT_TYPE determines the behavior of the NVCC compilation unit:
  1:     NVCC compiler, Unified Memory allocation, Parallel loops on GPUs   
  2:     NVCC compiler, Unified Memory allocation, Sequential loops on host
  Other: NVCC compiler, Standard heap allocation, Sequential loops on host  */
#define PFCUDA_COMP_UNIT_TYPE 1
  
/* extern "C" is required for the C source files when compiled with NVCC */

extern "C"{
  #include "overlandflow_eval_Kin.c"
}
  