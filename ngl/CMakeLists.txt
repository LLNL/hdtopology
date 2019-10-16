FIND_PATH( NGL_INCLUDE_DIR ngl.h
           PATHS "include" "../include")
find_package(ANN REQUIRED)

add_subdirectory(src)

if(ENABLE_PYTHON)
add_subdirectory(python)
endif()
