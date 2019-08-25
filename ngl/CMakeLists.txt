FIND_PATH( NGL_INCLUDE_DIR ngl.h
           PATHS "include" "../include")
find_package(ann REQUIRED)

add_subdirectory(python)
