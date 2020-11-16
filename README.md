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

Ninja 1.8.2
```

To use matchers/builders

``` 
clang -O3 -mllvm -polly -mllvm -polly-enable-matchers-opt-late -mllvm -polly-tile-for-cim -mllvm -debug-only=polly-opt-isl,polly-ast -I utilities/ -I linear-algebra/kernels/gemm/ utilities/polybench.c linear-algebra/kernels/gemm/gemm.c  -o gemm -ldl -L/pathTo/llvm_build/lib -lCIMRuntime 
```


How to install
```
export BASE=`pwd`
export LLVM_SRC=${BASE}/llvm
export POLLY_SRC=${LLVM_SRC}/tools/polly
export CLANG_SRC=${LLVM_SRC}/tools/clang
export LLVM_BUILD=${BASE}/llvm_build

set -e

if [ -e /proc/cpuinfo ]; then
    procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    procs=1
fi

if ! test -d ${LLVM_SRC}; then
    git clone http://llvm.org/git/llvm.git ${LLVM_SRC}
    (cd ${LLVM_SRC} && git reset --hard origin/release_80)
fi

if ! test -d ${POLLY_SRC}; then
    git clone --branch=master --single-branch https://github.com/LoopTactics/tdo-cim.git llvm-polly ${POLLY_SRC}
fi

if ! test -d ${CLANG_SRC}; then
    git clone http://llvm.org/git/clang.git ${CLANG_SRC}
    (cd ${CLANG_SRC} && git reset --hard origin/release_80)
fi

mkdir -p ${LLVM_BUILD}
cd ${LLVM_BUILD}

cmake ${LLVM_SRC}
make -j$procs -l$procs
make check-polly
```
