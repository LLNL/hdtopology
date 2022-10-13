FIND_PATH( NGL_INCLUDE_DIR ngl.h
           PATHS "include" "../include")
# find_package(ANN REQUIRED)

add_subdirectory(src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../external/annlibs/include)

IF (ENABLE_PYTHON)
    add_subdirectory(python)
ENDIF()
