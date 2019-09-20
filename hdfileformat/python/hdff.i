%module hdff

%include "typemaps.i"

%{
  #include "TypeConversion.h"
%}

%include "FileHandle.i"
%include "DatasetHandle.i"
%include "DataCollectionHandle.i"
%include "DataBlockHandle.i"
%include "DataPointsHandle.i"
%include "SegmentationHandle.i"
%include "Segmentation.i"
%include "HierarchicalSegmentation.i"
%include "MorseComplex.i"
%include "FunctionHandle.i"
%include "GraphHandle.i"
%include "TopoGraph.i"
%include "SubspaceHandle.i"
%include "ClusterHandle.i"
%include "BasisHandle.i"
%include "Basis.i"
%include "DistributionHandle.i"
%include "HistogramHandle.i"
