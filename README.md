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

Tested with: Polybench3.2 and 4.0

Tested with:
```
clang version 8.0.1 (http://llvm.org/git/clang.git 2e4c9c5fc864c2c432e4c262a67c42d824b265c6) (http://llvm.org/git/llvm.git ff8c1be17aa3ba7bacb1ef7dcdbecf05d5ab4eb7)
Target: x86_64-unknown-linux-gnu
Thread model: posix
```

To use matchers/builders

``` 
clang -O3 -mllvm -polly -mllvm -polly-enable-matchers-opt-early -mllvm -debug-only=polly-opt-isl -I utilities/ -I linear-algebra/kernels/gemm/ utilities/polybench.c linear-algebra/kernels/gemm/gemm.c  -o gemm -ldl -L/pathTo/llvm_build/lib -lCIMRuntime 
```
