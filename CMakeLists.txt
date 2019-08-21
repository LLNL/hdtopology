cmake_minimum_required(VERSION 3.12)

project(HDTopology)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(HDFileFormat)
find_package(NGLCu)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")


OPTION(ENABLE_PYTHON "Enable python binding" On)

ADD_SUBDIRECTORY(src)

IF (ENABLE_PYTHON)
    ADD_SUBDIRECTORY(python)
ENDIF()
