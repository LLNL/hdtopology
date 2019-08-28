find_package(NGLCu)

INCLUDE_DIRECTORIES(
        ${NGL_INCLUDE_DIR}
        ${ANN_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/topology/src
        ${CMAKE_SOURCE_DIR}/hdfileformat/src
)

if (NGLCU_FOUND)
    INCLUDE_DIRECTORIES(${NGLCU_INCLUDE_DIR})
    if (FLANN_INCLUDE_DIR)
	    INCLUDE_DIRECTORIES("${FLANN_INCLUDE_DIR}")
    endif()
    if (FAISS_INCLUDE_DIR)
	    INCLUDE_DIRECTORIES("${FAISS_INCLUDE_DIR}")
    endif()
    ADD_EXECUTABLE(testSummaryTopology testSummaryTopology.cpp)
    TARGET_LINK_LIBRARIES(testSummaryTopology ${ANN_LIBRARY} ${NGLCU_LIBRARY} chdtopology hdfileformat)
else()
  ADD_EXECUTABLE(testSummaryTopology testSummaryTopology.cpp)
  TARGET_LINK_LIBRARIES(testSummaryTopology ${ANN_LIBRARY} chdtopology hdfileformat)
endif()
