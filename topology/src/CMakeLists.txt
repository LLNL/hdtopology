if(ENABLE_CUDA)
  find_package(CUDA)
endif(ENABLE_CUDA)

SET (HDT_HEADERS
  EdgeIterator.h
  HDData.h
  JointDistributions.h
  Neighborhood.h
  ExtremumGraph.h
  Flags.h
  Histogram.h
  NeighborhoodIterator.h
  Selectivity.h
)

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
)

add_library(chdtopology STATIC ${HDT_SOURCES} ${HDT_HEADERS})
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
