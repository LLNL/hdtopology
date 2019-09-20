#streaming ngl
SET(NGL_SRC
        getNeighborGraph.cpp
        ANNSearchIndex.cpp
)

if(FAISS_FOUND)
list(APPEND NGL_SRC FLANNSearchIndex.cpp)
endif(FAISS_FOUND)

if(FLANN_FOUND)
list(APPEND NGL_SRC FAISSSearchIndex.cpp)
endif(FLANN_FOUND)

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${NGL_INCLUDE_DIR}
        ${ANN_INCLUDE_DIR}
)

################ compile and link ###################
if(ENABLE_CUDA)
  add_library(cngl STATIC ${NGL_SRC} ngl_cuda.cu Graph.cpp)
  set_target_properties(cngl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  link_libraries(cngl ${ANN_LIBRARY} ${CUDA_LIBRARIES})
else()
  add_library(cngl STATIC ${NGL_SRC})
  link_libraries(cngl ${ANN_LIBRARY})
endif(ENABLE_CUDA)

set_target_properties(cngl PROPERTIES POSITION_INDEPENDENT_CODE ON)
# SWIG_LINK_LIBRARIES(ngl ${Python3_LIBRARIES} ${ANN_LIBRARY})
