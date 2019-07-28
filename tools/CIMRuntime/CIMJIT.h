#ifndef CIMJIT_H_
#define CIMJIT_H_

void hello_cim_lib(int);
void cim_init(float);
void cim_tear_down(float);
void cim_allocate_shared_memory(float);
void cim_gemm_int(
int m, int n, int k, int *A, int lda, int *B, int ldb, int *C, int ldc);

#endif
