
SET (HDT_HEADERS
  EdgeIterator.h
  HDData.h
  JointDistributions.h
  Neighborhood.h
  ExtremumGraph.h
  Flags.h
  Histogram.h
  NGLIterator.h
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

if (NGLCU_FOUND)
  include_directories(${NGLCU_INCLUDE_DIR})
endif(NGLCU_FOUND)


ADD_LIBRARY(chdtopology STATIC ${HDT_SOURCES} ${HDT_HEADERS})
target_link_libraries(chdtopology hdfileformat)

if (NGLCU_FOUND)
  target_link_libraries(chdtopology ${NGLCU_LIBRARY})
endif (NGLCU_FOUND)

INSTALL(FILES ${HDT_HEADERS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/HDTopology
)

INSTALL(TARGETS chdtopology
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
