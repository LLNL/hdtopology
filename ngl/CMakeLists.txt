FIND_PATH( NGL_INCLUDE_DIR ngl.h
           PATHS "include" "../include")
# find_package(ANN REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../external/annlibs/include)

add_subdirectory(src)

if(ENABLE_PYTHON)
add_subdirectory(python)
endif()
