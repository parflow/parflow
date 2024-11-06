

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void* create_random_tensor(int rows, int cols);
void print_tensor(void* tensor_ptr);
void free_tensor(void* tensor_ptr);

#ifdef __cplusplus
}
#endif


