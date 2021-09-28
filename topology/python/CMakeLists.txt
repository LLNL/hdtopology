cmake_minimum_required(VERSION 3.12)

#cmake_policy(SET CMP0086 NEW)
cmake_policy(SET CMP0078 NEW)

find_package(SWIG REQUIRED)

INCLUDE(${SWIG_USE_FILE})

set(CMAKE_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Development NumPy)

set(CMAKE_VERBOSE_MAKEFILE YES)

set(SWIG_SRC
       hdtopology.i
)

MESSAGE("PYTHON_EXECUTABLE: " ${Python3_EXECUTABLE})
MESSAGE("PYTHON_INCLUDE_DIR: " ${Python3_INCLUDE_DIRS})
MESSAGE("PYTHON_LIBRARIES: " ${Python3_LIBRARIES})
MESSAGE("PYTHON_NUMPY_INCLUDE: " ${Python3_NumPy_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/topology/src
        ${CMAKE_SOURCE_DIR}/topology/python
        ${CMAKE_SOURCE_DIR}/hdfileformat/src
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        ${HDFILEFORMAT_INCLUDE_DIR}
)

if (NGLCU_FOUND)
  include_directories(${NGLCU_INCLUDE_DIR})
endif(NGLCU_FOUND)

set_property(SOURCE ${SWIG_SRC} PROPERTY CPLUSPLUS ON)

################ compile and link ###################
SWIG_ADD_LIBRARY(hdtopology TYPE MODULE LANGUAGE python SOURCES ${SWIG_SRC})
SWIG_LINK_LIBRARIES(hdtopology chdtopology)

if (APPLE)
  set_target_properties(hdtopology PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif (WIN32)
  target_link_libraries(hdtopology ${Python3_LIBRARIES})
elseif (LINUX)
  set_target_properties(hdtopology PROPERTIES LINK_FLAGS "-Wl,--unresolved-symbols=ignore-all")
endif()

if (NGLCU_FOUND)
  include_directories(${NGLCU_INCLUDE_DIR})
  swig_link_libraries(hdtopology ${NGLCU_LIBRARY})
endif(NGLCU_FOUND)

################ install ###################

SET(PKG_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hdtopology.py
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    )

INSTALL(FILES ${PKG_FILES}
        DESTINATION ${Python3_SITEARCH}/hdtopology
)

INSTALL(TARGETS hdtopology
        DESTINATION ${Python3_SITEARCH}/hdtopology
)

if(ENABLE_PACKAGING)
    install(FILES ${PKG_FILES}
            DESTINATION ${CMAKE_SOURCE_DIR}/deploy/hdtopology/hdtopology
    )

    install(TARGETS hdtopology
            DESTINATION ${CMAKE_SOURCE_DIR}/deploy/hdtopology/hdtopology
    )
endif()
