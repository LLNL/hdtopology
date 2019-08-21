# This will define:
# NGLCU_FOUND - system has NGLCu
# NGLCU_INCLUDE_DIR - the NGLCu include directory
# NGLCU_LIBRARY - Link these to use NGLCu

find_path(NGLCU_INCLUDE_DIR ngl_cuda.h PATHS "${CMAKE_SOURCE_DIR}/../../nglcu/include" "${CMAKE_SOURCE_DIR}/external/nglcu/include" NO_DEFAULT_PATH)
find_library(NGLCU_LIBRARY NAMES nglcu PATHS "${CMAKE_SOURCE_DIR}/../../nglcu/" "${CMAKE_SOURCE_DIR}/external/nglcu")

if(NGLCU_INCLUDE_DIR AND NGLCU_LIBRARY)
	  set(NGLCU_FOUND 1)
  else()
	    set(NGLCU_FOUND 0)
    endif()

    if(NGLCU_FOUND)
	      message(STATUS "Found NGLCu library: ${NGLCU_LIBRARY}")
      endif()

