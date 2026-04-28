# MUSA backend: toolchain, source files, and build configuration.
if(NOT USE_MUSA)
  return()
endif()

if(DEFINED ENV{MUSA_HOME})
  set(MUSA_HOME $ENV{MUSA_HOME})
else()
  set(MUSA_HOME "/usr/local/musa")
endif()
list(APPEND CMAKE_MODULE_PATH ${MUSA_HOME}/cmake)
find_package(MUSAToolkit REQUIRED)
if(NOT MUSAToolkit_FOUND)
  message(FATAL_ERROR "Cannot find MUSA, USE_MUSA=" ${USE_MUSA})
endif()
message(STATUS "Build with MUSA ${MUSAToolkit_VERSION} support")

add_compile_definitions("MUSA_MAJOR_VERSION=${MUSAToolkit_VERSION_MAJOR}")

file(GLOB TILE_LANG_MUSA_SRCS
  src/runtime/runtime.cc
  src/target/codegen_musa.cc
  src/target/ptx.cc
  src/target/rt_mod_musa.cc
)

list(APPEND TILE_LANG_SRCS ${TILE_LANG_MUSA_SRCS})
list(APPEND TILE_LANG_INCLUDES ${MUSAToolkit_INCLUDE_DIRS})
