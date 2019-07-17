#include "CIMJIT.h"
#include <stdio.h>

#define dump_function() printf("-> %s\n", __func__)

void hello_cim_lib(int i) { dump_function(); }
