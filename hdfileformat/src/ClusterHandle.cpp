#include "ClusterHandle.h"

namespace HDFileFormat{

const std::string ClusterHandle::sDefaultClusterName = "FlatCluster";

ClusterHandle::ClusterHandle(HandleType t)
 :DataBlockHandle(t)
{
  mID = sDefaultClusterName;
}

ClusterHandle::ClusterHandle(const char *filename, HandleType t)
 :DataBlockHandle(filename,t)
{
  mID = sDefaultClusterName;
}

//copy constructor
ClusterHandle::ClusterHandle(const ClusterHandle& handle)
 :DataBlockHandle(handle)
{
}

ClusterHandle::~ClusterHandle()
{
  //printf("ClusterHandle destructor: %s\n", mID.c_str());
}

ClusterHandle& ClusterHandle::operator=(const ClusterHandle& handle)
{
  DataBlockHandle::operator=(handle);

  return *this;
}

FileHandle &ClusterHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_DATABLOCK: //why datablock can attach to ClusterHandle?
      return FileHandle::add(handle);
      break;
    case H_SUBSPACE:
      return FileHandle::add(handle);
      break;
    case H_HIERARCHY:
      return FileHandle::add(handle);
      break;
    case H_CLUSTER:
    case H_EMBEDDING:
    case H_COLLECTION:
    case H_BASIS:
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside cluster.",handle.typeName());
      break;
  }

  return *this;
}

void ClusterHandle::setLabel(uint32_t *label, int pointCount)
{
  DataBlockHandle::setData( label, pointCount, 1 );
}

//FIXME need better type check
ClusteringResultType ClusterHandle::GetClusteringResultType()
{
  SubspaceHandle subspace;
  this->getFirstChildByType(subspace);
  if(subspace.isValid())
    return ClusterType_FlatCluster_Subspace;
  else
    return ClusterType_FlatCluster;
}

void ClusterHandle::readHierarchy(HDFileFormat::ExplicitHierarchy &hierarchy)
{

}

int ClusterHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);
  getAttribute(node, "ClusterParameters", mClusteringParameterString);

  return 1;
}

int ClusterHandle::attachXMLInternal(XMLNode &node) const
{
  DataBlockHandle::attachXMLInternal(node);
  addAttribute(node, "ClusterParameters", mClusteringParameterString);

  return 1;
}

}
