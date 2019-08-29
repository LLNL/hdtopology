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
    ADD_EXECUTABLE(testReadSummaryTopology testReadSummaryTopology.cpp)
    TARGET_LINK_LIBRARIES(testReadSummaryTopology ${ANN_LIBRARY} ${NGLCU_LIBRARY} chdtopology hdfileformat)

    ADD_EXECUTABLE(testWriteSummaryTopology testWriteSummaryTopology.cpp)
    TARGET_LINK_LIBRARIES(testWriteSummaryTopology ${ANN_LIBRARY} ${NGLCU_LIBRARY} chdtopology hdfileformat)
else()

  ADD_EXECUTABLE(testReadSummaryTopology testReadSummaryTopology.cpp)
  TARGET_LINK_LIBRARIES(testReadSummaryTopology ${ANN_LIBRARY} chdtopology hdfileformat)

  ADD_EXECUTABLE(testWriteSummaryTopology testWriteSummaryTopology.cpp)
  TARGET_LINK_LIBRARIES(testWriteSummaryTopology ${ANN_LIBRARY} chdtopology hdfileformat)
endif()
