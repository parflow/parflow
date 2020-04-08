/* PF_COMP_UNIT_TYPE determines the behavior of the NVCC compilation unit and/or OpenMP loops:
	 ------------------------------------------------------------
   CUDA
	 ------------------------------------------------------------
	 1:     NVCC compiler, Unified Memory allocation, Parallel loops on GPUs
	 2:     NVCC compiler, Unified Memory allocation, Sequential loops on host
	 Other: NVCC compiler, Standard heap allocation, Sequential loops on host


	 ------------------------------------------------------------
   OpenMP
	 ------------------------------------------------------------
   1:     CXX compiler, Unified Memory allocation, Parallel loops on CPU
	 2:     CXX compiler, Unified Memory allocation, Sequential loops on CPU
	 Other: CXX compiler, Standard heap allocation, Sequential loops on CPU
*/
#define PF_COMP_UNIT_TYPE 1

/* extern "C" is required for the C source files when compiled with NVCC */

extern "C"{
  #include "problem_phase_rel_perm.c"
}
