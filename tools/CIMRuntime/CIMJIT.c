#include "CIMJIT.h"
#include <stdio.h>
#include <unistd.h>

#define dump_function() printf("-> %s\n", __func__)

void hello_cim_lib(int i) { dump_function(); }
void cim_init(float dummy) { dump_function(); }
void cim_tear_down(float dummy) { dump_function(); }
void cim_allocate_shared_memory(float bytes) {
  dump_function();
  printf("bytes: %f\n", bytes);
}
void cim_gemm_int(int m, int n, int k, int *A, int lda, int *B, int ldb, int *C,
                  int ldc) {

  printf("%s\n", "param:");
  printf("%s%d\n", "m: ", m);
  printf("%s%d\n", "n: ", n);
  printf("%s%d\n", "k: ", k);

  dump_function();
  sleep(1);
}

void cim_gemv_int(int m, int n, int *A, int lda, int *X, int inc_x, int *Y,
                  int inc_y) {

  printf("%s\n", "param:");
  printf("%s%d\n", "m: ", m);
  printf("%s%d\n", "n: ", n);

  dump_function();
  sleep(1);
}
