# MUSA target helpers are pure C++ registrations used by common TileLang code.
# Keep them available even when the MUSA toolchain is disabled.
list(APPEND TILE_LANG_SRCS src/musa/target_utils.cc)

if(NOT USE_MUSA)
  return()
endif()

message(STATUS "MUSA Backend is enabled")

list(APPEND TILE_LANG_SRCS
  src/musa/codegen/codegen_musa.cc
  src/musa/codegen/intrin_rule_musa.cc
)
