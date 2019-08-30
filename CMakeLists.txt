cmake_minimum_required(VERSION 3.12)

project(hdtopology)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")
set(CMAKE_CXX_STANDARD 11)

find_package(NGLCu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

if(WIN32)
  set(CMAKE_SHARED_LINKER_FLAGS "/FORCE:MULTIPLE")
  set(CMAKE_MODULE_LINKER_FLAGS "/FORCE:MULTIPLE")
  set(CMAKE_EXE_LINKER_FLAGS "/FORCE:MULTIPLE")
endif()

option(ENABLE_PYTHON "Enable python binding" On)

add_subdirectory(hdfileformat)
add_subdirectory(ngl)
add_subdirectory(topology)
add_subdirectory(utilities)
