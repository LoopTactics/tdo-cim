if (CUDA_FOUND OR OpenCL_FOUND)
  add_subdirectory(GPURuntime)
endif (CUDA_FOUND OR OpenCL_FOUND)

if (CIM_RUNTIME)
  add_subdirectory(CIMRuntime)
  message(STATUS "Adding CIMRuntime library")
endif (CIM_RUNTIME)

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
