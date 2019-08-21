# Try to find HDFileFormat library.
# This will define:
# HDFILEFORMAT_FOUND - system has hdfileformat
# HDFILEFORMAT_INCLUDE_DIR - the hdfileformat include directory
# HDFILEFORMAT_LIBRARY - Link these to use hdfileformat

cmake_minimum_required(VERSION 3.12)

find_path(HDFILEFORMAT_INCLUDE_DIR HDFileFormat/FileHandle.h PATHS "${CMAKE_SOURCE_DIR}/../hdfileformat/build/include")
find_library(HDFILEFORMAT_LIBRARY NAMES HDFileFormat PATHS "${CMAKE_SOURCE_DIR}/../hdfileformat/build/lib")

# message(${CMAKE_SOURCE_DIR}/../../)

if(HDFILEFORMAT_INCLUDE_DIR AND HDFILEFORMAT_LIBRARY)
  set(HDFILEFORMAT_FOUND 1)
else()
  set(HDFILEFORMAT_FOUND 0)
endif()

if(HDFILEFORMAT_FOUND)
  message(STATUS "Found HDFileFormat library: ${HDFILEFORMAT_LIBRARY}")
endif()
