# MUSA backend: toolchain, stub libraries, source files, and build configuration.
if(NOT USE_MUSA)
  return()
endif()

message(STATUS "MUSA Backend is enabled")

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

if(TILELANG_USE_MUSA_STUBS)
  if(WIN32 AND NOT CYGWIN)
    message(FATAL_ERROR "TILELANG_USE_MUSA_STUBS=ON is not supported on Windows. "
                        "Please configure with -DTILELANG_USE_MUSA_STUBS=OFF.")
  endif()

  add_library(musa_stub SHARED src/backend/musa/stubs/musa.cc)
  target_include_directories(musa_stub PRIVATE ${MUSAToolkit_INCLUDE_DIRS})
  target_compile_definitions(musa_stub PRIVATE TILELANG_MUSA_STUB_EXPORTS)
  target_link_libraries(musa_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(musa_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "musa_stub"
  )

  add_library(musart_stub SHARED src/backend/musa/stubs/musart.cc)
  target_include_directories(musart_stub PRIVATE ${MUSAToolkit_INCLUDE_DIRS})
  target_link_libraries(musart_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(musart_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "musart_stub"
  )

  set(TILELANG_MUSA_STUB_LIBRARY "musart_stub;musa_stub" CACHE INTERNAL
      "MUSA runtime and driver stub libraries")
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK musa_stub musart_stub)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_TARGETS musa_stub musart_stub)
endif()

add_compile_definitions("MUSA_MAJOR_VERSION=${MUSAToolkit_VERSION_MAJOR}")

file(GLOB TILE_LANG_MUSA_SRCS
  src/runtime/runtime.cc
  src/backend/musa/codegen/ptx.cc
  src/backend/musa/codegen/codegen_musa.cc
  src/backend/musa/codegen/rt_mod_musa.cc
  src/backend/musa/op/*.cc
)

list(REMOVE_ITEM TILE_LANG_MUSA_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/musa/op/copy_analysis.cc")
list(APPEND TILE_LANG_SRCS ${TILE_LANG_MUSA_SRCS})
list(APPEND TILE_LANG_INCLUDES ${MUSAToolkit_INCLUDE_DIRS})
link_directories(${MUSAToolkit_LIBRARY_DIR})
