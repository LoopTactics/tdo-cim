set(MODULE TRUE)
set(LLVM_NO_RTTI 1)

add_polly_library(CIMRuntime
  CIMJIT.c
  )

#FIXME: can we avoid c and use c++ instead?

set_target_properties(CIMRuntime
  PROPERTIES
  LINKER_LANGUAGE C
  PREFIX "lib"
  )

set_property(TARGET CIMRuntime PROPERTY C_STANDARD 99)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default ")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-sanitize=all ")
endif()

