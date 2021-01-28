#streaming ngl
SET(NGL_SRC
        getNeighborGraph.cpp
        ANNSearchIndex.cpp
        Graph.cpp
)

set(NGL_INC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NGL_INCLUDE_DIR}
    ${ANN_INCLUDE_DIR}
)

set(NGL_LINK_LIB
    ${ANN_LIBRARY}
)

if(ENABLE_CUDA)
  ADD_DEFINITIONS(-DENABLE_CUDA)
endif(ENABLE_CUDA)

if(ENABLE_OPENMP)
  ADD_DEFINITIONS(-DENABLE_OPENMP)
endif(ENABLE_OPENMP)


if(FAISS_FOUND)
  list(APPEND NGL_SRC FAISSSearchIndex.cpp)
  list(APPEND NGL_INC ${FAISS_INCLUDE_DIR})
  list(APPEND NGL_LINK_LIB ${FAISS_LIBRARY})
endif(FAISS_FOUND)

if(FLANN_FOUND)
  list(APPEND NGL_SRC FLANNSearchIndex.cpp)
  list(APPEND NGL_INC ${FLANN_INCLUDE_DIR})
  list(APPEND NGL_INC ${LZ4_INCLUDE_DIR})
  list(APPEND NGL_LINK_LIB ${FLANN_LIBRARY})
endif(FLANN_FOUND)

INCLUDE_DIRECTORIES(
    ${NGL_INC}
)

################ compile and link ###################
if(ENABLE_CUDA)
  add_library(cngl STATIC ${NGL_SRC} ngl_cuda.cu )
  set_target_properties(cngl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  link_libraries(cngl ${NGL_LINK_LIB} ${CUDA_LIBRARIES})
else()
  add_library(cngl STATIC ${NGL_SRC})
  link_libraries(cngl ${NGL_LINK_LIB})
endif(ENABLE_CUDA)

if(ENABLE_OPENMP)
  target_link_libraries(cngl PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(cngl PROPERTIES POSITION_INDEPENDENT_CODE ON)
# SWIG_LINK_LIBRARIES(ngl ${Python3_LIBRARIES} ${ANN_LIBRARY})
