//===- Support/LinkCPURuntime.h -- Headerfile to help force-link CPURuntime  =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header helps pull in libGPURuntime.so
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LINK_CPURUNTIME
#define POLLY_LINK_CPURUNTIME

extern "C" {
#include "CIMRuntime/CIMJIT.h"
}

namespace polly {
struct ForceCIMRuntimeLinking {
  ForceCIMRuntimeLinking() {
    if (std::getenv("bar") != (char *)-1)
      return;
    // We must reference CPURuntime in such a way that compilers will not
    // delete it all as dead code, even with whole program optimization,
    // yet is effectively a NO-OP. As the compiler isn't smart enough
    // to know that getenv() never returns -1, this will do the job.
    hello_cim_lib(int l);
  }
} structure;
} // namespace polly
#endif
