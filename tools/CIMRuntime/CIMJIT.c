#include "CIMJIT.h"
#include <stdio.h>
#include <unistd.h>

#define dump_function() printf("-> %s\n", __func__)

void hello_cim_lib(int i) { dump_function(); }
void cim_init(float dummy) { dump_function(); }
void cim_tear_down(float dummy) { dump_function(); }
void cim_allocate_shared_memory(float bytes) { 
  dump_function(); printf("bytes: %f\n", bytes); 
}
void cim_gemm_double(double *A, double *B, double *C) { dump_function(); sleep(5); } 
