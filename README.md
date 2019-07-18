Polly - Polyhedral optimizations for LLVM
-----------------------------------------
http://polly.llvm.org/

Polly uses a mathematical representation, the polyhedral model, to represent and
transform loops and other control flow structures. Using an abstract
representation it is possible to reason about transformations in a more general
way and to use highly optimized linear programming libraries to figure out the
optimal loop structure. These transformations can be used to do constant
propagation through arrays, remove dead loop iterations, optimize loops for
cache locality, optimize arrays, apply advanced automatic parallelization, drive
vectorization, or they can be used to do software pipelining.

To use matchers/builders

``` 
clang -O3 -mllvm -polly -mllvm -polly-enable-matchers-opt-early -mllvm -debug-only=polly-opt-isl -I utilities/ -I linear-algebra/kernels/gemm/ utilities/polybench.c linear-algebra/kernels/gemm/gemm.c  -o gemm -ldl -L/pathTo/llvm_build/lib -lCIMRuntime 
```
