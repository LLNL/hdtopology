cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

option(ENABLE_PYTHON "Enable python binding" On)
option(ENABLE_STREAMING "Enable streaming topology computation" On)
option(ENABLE_PACKAGING "Enable packaging for PyPI" Off)
option(ENABLE_CUDA "Enable GPU graph pruning" Off)

option(ENABLE_UTILTIES "Compile c++ utilities" ON)

option(ENABLE_FLANN "Enable FLANN for neighborhood lookup" Off)
option(ENABLE_OPENMP "Enable OpenMP for FLANN" Off)

if(ENABLE_CUDA)
  project(hdtopology CXX CUDA)
else(ENABLE_CUDA)
  project(hdtopology CXX)
endif(ENABLE_CUDA)

if(ENABLE_FLANN)
    find_package(FLANN REQUIRED)
    find_package(LZ4 REQUIRED)
endif()

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(WIN32)
  set(CMAKE_SHARED_LINKER_FLAGS "/FORCE:MULTIPLE")
  set(CMAKE_MODULE_LINKER_FLAGS "/FORCE:MULTIPLE")
  set(CMAKE_EXE_LINKER_FLAGS "/FORCE:MULTIPLE")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()


add_subdirectory(hdfileformat)
add_subdirectory(ngl)
add_subdirectory(external/annlibs)
add_subdirectory(topology)
add_subdirectory(utilities)
