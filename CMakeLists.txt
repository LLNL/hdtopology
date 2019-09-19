cmake_minimum_required(VERSION 3.12)

project(hdtopology)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")
set(CMAKE_CXX_STANDARD 11)

if(WIN32)
  set(CMAKE_SHARED_LINKER_FLAGS "/FORCE:MULTIPLE")
  set(CMAKE_MODULE_LINKER_FLAGS "/FORCE:MULTIPLE")
  set(CMAKE_EXE_LINKER_FLAGS "/FORCE:MULTIPLE")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

option(ENABLE_PYTHON "Enable python binding" On)
option(ENABLE_PACKAGING "Enable packaging for PyPI" Off)
option(ENABLE_CUDA "Enable GPU graph pruning" Off)

add_subdirectory(hdfileformat)
add_subdirectory(ngl)
add_subdirectory(topology)
add_subdirectory(utilities)
