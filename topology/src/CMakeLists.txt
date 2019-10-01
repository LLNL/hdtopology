if(ENABLE_CUDA)
  find_package(CUDA)
endif(ENABLE_CUDA)

SET (HDT_SOURCES
 ExtremumGraph.cpp
 HDData.cpp
 Histogram.cpp
 JointDistributions.cpp
 Neighborhood.cpp
 Selectivity.cpp
 )

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/external
  ${CMAKE_SOURCE_DIR}/hdfileformat/src
  ${CMAKE_SOURCE_DIR}/ngl/include
)

add_library(chdtopology STATIC ${HDT_SOURCES})
target_link_libraries(chdtopology hdfileformat)
set_target_properties(chdtopology PROPERTIES POSITION_INDEPENDENT_CODE ON)


########### for using hdtopology as a c++ library ###########
# INSTALL(FILES ${HDT_HEADERS}
#         DESTINATION ${CMAKE_INSTALL_PREFIX}/include/HDTopology
# )
#
# INSTALL(TARGETS chdtopology
#         DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# )
