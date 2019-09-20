cmake_minimum_required(VERSION 3.12)
#cmake_policy(SET CMP0086 NEW)
cmake_policy(SET CMP0078 NEW)

FIND_PACKAGE(SWIG REQUIRED)

INCLUDE(${SWIG_USE_FILE})

set(CMAKE_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

SET(SWIG_SRC
       ngl.i
)

INCLUDE_DIRECTORIES(
        ${NGL_INCLUDE_DIR}
        ${ANN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

set_property(SOURCE ${SWIG_SRC} PROPERTY CPLUSPLUS ON)

################ compile and link ###################
if(ENABLE_CUDA)
  SWIG_ADD_LIBRARY(ngl TYPE MODULE LANGUAGE python SOURCES ${SWIG_SRC})
  SWIG_LINK_LIBRARIES(ngl cngl ${ANN_LIBRARY} ${CUDA_LIBRARIES})
else()
  SWIG_ADD_LIBRARY(ngl TYPE MODULE LANGUAGE python SOURCES ${SWIG_SRC})
  SWIG_LINK_LIBRARIES(ngl cngl ${ANN_LIBRARY})
endif(ENABLE_CUDA)

set_target_properties(ngl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (APPLE)
  set_target_properties(ngl PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif (WIN32)
  target_link_libraries(ngl ${Python3_LIBRARIES})
elseif (LINUX)
  set_target_properties(ngl PROPERTIES LINK_FLAGS "-Wl,--unresolved-symbols=ignore-all")
endif()

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
