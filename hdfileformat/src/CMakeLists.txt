
SET (HD_FILE_FORMAT_HEADERS
    Definitions.h
    HDFileFormatUtility.h
    FileHandle.h
    FileData.h
    DataCollectionHandle.h
    #GroupHandle.h

    DatasetHandle.h
    DataBlockHandle.h
    DataPointsHandle.h
    DataPointsMetaInfoHandle.h

    ClusterHandle.h
    EmbeddingHandle.h
    HierarchyHandle.h
    SubspaceHandle.h
    BasisHandle.h
    GraphHandle.h
    SegmentationHandle.h
    FunctionHandle.h

    ExplicitHierarchy.h
    Segmentation.h
    HierarchicalSegmentation.h
    MorseComplex.h
    TopoGraph.h
    Basis.h

    # ParameterParser.h

    xmlParser.h

    PointSetMetaInfo.h
    DistributionHandle.h
    HistogramHandle.h
    #ThinFileIO.h
)

SET (HD_FILE_FORMAT_SOURCES
    FileHandle.cpp
    FileData.cpp
    DataCollectionHandle.cpp
    #GroupHandle.cpp

    DatasetHandle.cpp
    DataBlockHandle.cpp
    DataPointsHandle.cpp
    DataPointsMetaInfoHandle.cpp

    ClusterHandle.cpp
    EmbeddingHandle.cpp
    HierarchyHandle.cpp
    SubspaceHandle.cpp
    BasisHandle.cpp
    GraphHandle.cpp
    SegmentationHandle.cpp
    FunctionHandle.cpp

    ExplicitHierarchy.cpp
    Segmentation.cpp
    HierarchicalSegmentation.cpp
    MorseComplex.cpp
    TopoGraph.cpp

    # ParameterParser.cpp
    xmlParser.cpp

    PointSetMetaInfo.cpp

    DistributionHandle.cpp
    HistogramHandle.cpp
    #ThinFileIO.cpp
)

INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_LIBRARY(hdfileformat STATIC ${HD_FILE_FORMAT_SOURCES} ${HD_FILE_FORMAT_HEADERS})
