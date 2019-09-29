if(ENABLE_FLANN)
  ADD_DEFINITIONS(-DENABLE_FLANN)
  INCLUDE_DIRECTORIES(${FLANN_INCLUDE_DIR})
endif(ENABLE_FLANN)

if(ENABLE_FAISS)
  ADD_DEFINITIONS(-DENABLE_FAISS)
endif(ENABLE_FAISS)

IF(ENABLE_STREAMING)
  ADD_DEFINITIONS(-DENABLE_STREAMING)
ENDIF(ENABLE_STREAMING)

INCLUDE_DIRECTORIES(
        ${NGL_INCLUDE_DIR}
        ${ANN_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/topology/src
        ${CMAKE_SOURCE_DIR}/hdfileformat/src
)

if (ENABLE_CUDA)
    INCLUDE_DIRECTORIES(${NGLCU_INCLUDE_DIR})
    if (FLANN_INCLUDE_DIR)
	    INCLUDE_DIRECTORIES("${FLANN_INCLUDE_DIR}")
    endif()
    if (FAISS_INCLUDE_DIR)
	    INCLUDE_DIRECTORIES("${FAISS_INCLUDE_DIR}")
    endif()
    ADD_EXECUTABLE(testReadSummaryTopology testReadSummaryTopology.cpp)
    TARGET_LINK_LIBRARIES(testReadSummaryTopology ${ANN_LIBRARY} ${CUDA_LIBRARIES} chdtopology hdfileformat)

    ADD_EXECUTABLE(testWriteSummaryTopology testWriteSummaryTopology.cpp)
    TARGET_LINK_LIBRARIES(testWriteSummaryTopology ${ANN_LIBRARY} ${CUDA_LIBRARIES} chdtopology hdfileformat)

    # ADD_EXECUTABLE(testWriteStreamingTopology testWriteStreamingTopology.cpp)
    # TARGET_LINK_LIBRARIES(testWriteStreamingTopology ${ANN_LIBRARY} ${CUDA_LIBRARIES} chdtopology hdfileformat)
else()

  ADD_EXECUTABLE(testReadSummaryTopology testReadSummaryTopology.cpp)
  TARGET_LINK_LIBRARIES(testReadSummaryTopology ${ANN_LIBRARY} chdtopology hdfileformat)

  ADD_EXECUTABLE(testWriteSummaryTopology testWriteSummaryTopology.cpp)
  TARGET_LINK_LIBRARIES(testWriteSummaryTopology ${ANN_LIBRARY} chdtopology hdfileformat cngl)
endif()

if(ENABLE_FLANN)
  TARGET_LINK_LIBRARIES(testWriteSummaryTopology ${LZ4_LIBRARY} ${FLANN_LIBRARY})
endif()

SET(INSTALL_FILES
      ${CMAKE_SOURCE_DIR}/utilities/testReadSummaryTopology.py
      ${CMAKE_SOURCE_DIR}/utilities/testWriteSummaryTopology.py
      ${CMAKE_SOURCE_DIR}/deploy/deployConfig.py
)

INSTALL(FILES ${INSTALL_FILES}
      DESTINATION ${CMAKE_BINARY_DIR}
)
INSTALL(TARGETS testWriteSummaryTopology  testReadSummaryTopology
    DESTINATION ${CMAKE_BINARY_DIR}
)
