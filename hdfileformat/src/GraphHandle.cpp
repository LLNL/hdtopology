#include "GraphHandle.h"

namespace HDFileFormat{

const std::string GraphHandle::sDefaultGraphName = "NeighborhoodGraph";

GraphHandle::GraphHandle(HandleType t)
  :DataBlockHandle(t),
   mGraphType(GRAPH_STORAGE_UNDEFINED)
{
  mID = sDefaultGraphName;
}

GraphHandle::GraphHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename,t),
   mGraphType(GRAPH_STORAGE_UNDEFINED)
{
  mID = sDefaultGraphName;
}

GraphHandle::GraphHandle(const GraphHandle& handle)
  :DataBlockHandle(handle),
   mGraphType(handle.mGraphType)
{
}

GraphHandle::~GraphHandle()
{
}

GraphHandle& GraphHandle::operator=(const GraphHandle& handle)
{
  DataBlockHandle::operator=(handle);
  mGraphType = handle.mGraphType;

  return *this;
}

FileHandle &GraphHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_DATABLOCK: //store graph weight
      return FileHandle::add(handle);
      break;
    case H_SUBSPACE:
    case H_HIERARCHY:
    case H_CLUSTER:
    case H_EMBEDDING:
    case H_COLLECTION:
    case H_BASIS:
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside graph.",handle.typeName());
      break;
  }

  return *this;
}

void GraphHandle::setEdgePairs(uint32_t *edgePair, uint32_t edgePairCount)
{
  //set the buffer
  DataBlockHandle::setData((void*)(edgePair));
  mValueSize = sizeof(uint32_t);
  mDimension = 2;
  mSampleCount = edgePairCount;
  mDataType = std::string("uint32_t");

  //set storage type
  mGraphType = GRAPH_STORAGE_TYPE_EDGE_PAIR;
}

void GraphHandle::setEdgeWeight(float *edgeWeight, int edgeCount)
{

}

int GraphHandle::parseXMLInternal(const XMLNode& node)
{
  DataBlockHandle::parseXMLInternal(node);

  return 1;
}

int GraphHandle::attachXMLInternal(XMLNode& node) const
{
  DataBlockHandle::attachXMLInternal(node);

  return 1;
}


}
