cmake_minimum_required(VERSION 3.12)
#cmake_policy(SET CMP0086 NEW)
cmake_policy(SET CMP0078 NEW)

FIND_PACKAGE(SWIG REQUIRED)

INCLUDE(${SWIG_USE_FILE})

# IF (NOT PYTHON_EXECUTABLE)
#    INCLUDE(FindPythonInterp)
# ENDIF (NOT PYTHON_EXECUTABLE)

set(CMAKE_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

# MESSAGE("ANN_INCLUDE_DIR: " ${ANN_INCLUDE_DIR})

# SET(CMAKE_SWIG_FLAGS "")
SET(SWIG_SRC
       ngl.i
)

SET(SWIG_CC_SRC
        getNeighborGraph.cpp
)

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${NGL_INCLUDE_DIR}
        ${ANN_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

set_property(SOURCE ${SWIG_SRC} PROPERTY CPLUSPLUS ON)

################ compile and link ###################
SWIG_ADD_LIBRARY(ngl TYPE MODULE LANGUAGE python SOURCES ${SWIG_SRC} ${SWIG_CC_SRC})
# target_compile_definitions(ngl PRIVATE Py_LIMITED_API)
SWIG_LINK_LIBRARIES(ngl ${Python3_LIBRARIES} ${ANN_LIBRARY})

############### install #################
SET(PKG_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ngl.py
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    )

INSTALL(FILES ${PKG_FILES}
        DESTINATION ${Python3_SITEARCH}/ngl
)

INSTALL(TARGETS ngl
        DESTINATION ${Python3_SITEARCH}/ngl
)

if(ENABLE_PACKAGING)
    install(FILES ${PKG_FILES}
            DESTINATION ${CMAKE_SOURCE_DIR}/deploy/ngl/ngl
    )

    install(TARGETS ngl
            DESTINATION ${CMAKE_SOURCE_DIR}/deploy/ngl/ngl
    )
endif()
