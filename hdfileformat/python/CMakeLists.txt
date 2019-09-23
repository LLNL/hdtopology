cmake_minimum_required(VERSION 3.12)
# cmake_policy(SET CMP0086 NEW)
cmake_policy(SET CMP0078 NEW)

FIND_PACKAGE(SWIG REQUIRED)

INCLUDE(${SWIG_USE_FILE})

set(CMAKE_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

# MESSAGE("PYTHON_EXECUTABLE: " ${Python3_EXECUTABLE})
# MESSAGE("PYTHON_INCLUDE_DIR: " ${Python3_INCLUDE_DIRS})
# MESSAGE("PYTHON_LIBRARIES: " ${Python3_LIBRARIES})
# MESSAGE("PYTHON_NUMPY_INCLUDE: " ${Python3_NumPy_INCLUDE_DIRS})


SET(SWIG_SRC
    hdff.i
)

INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/hdfileformat/src
        ${CMAKE_SOURCE_DIR}/hdfileformat/python
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(${SWIG_SRC} PROPERTIES CPLUSPLUS ON)

SWIG_ADD_LIBRARY(hdff TYPE MODULE LANGUAGE python SOURCES ${SWIG_SRC})
SWIG_LINK_LIBRARIES(hdff hdfileformat)

if (APPLE)
  set_target_properties(hdff PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif (WIN32)
  target_link_libraries(hdff ${Python3_LIBRARIES})
elseif (LINUX)
  set_target_properties(hdff PROPERTIES LINK_FLAGS "-Wl,--unresolved-symbols=ignore-all")
endif()

set(SWIG_MODULE_hdff_EXTRA_DEPS typemaps.i)


set(PKG_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hdff.py
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    )


install(FILES ${PKG_FILES}
        DESTINATION ${Python3_SITEARCH}/hdff
)

install(TARGETS hdff
        DESTINATION ${Python3_SITEARCH}/hdff
)

if(ENABLE_PACKAGING)
    install(FILES ${PKG_FILES}
            DESTINATION ${CMAKE_SOURCE_DIR}/deploy/hdff/hdff
    )

    install(TARGETS hdff
            DESTINATION ${CMAKE_SOURCE_DIR}/deploy/hdff/hdff
    )
endif()
