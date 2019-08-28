#include "DistributionHandle.h"

namespace HDFileFormat{

const std::string DistributionHandle::sDefaultName = "distribution";

DistributionHandle::DistributionHandle(HandleType t)
  :DataBlockHandle(t)
{
  mID = sDefaultName;
}

DistributionHandle::DistributionHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename, t)
{
  mID = sDefaultName;
}

DistributionHandle::DistributionHandle(const DistributionHandle &handle)
  :DataBlockHandle(handle)
{
  mID = sDefaultName;
}

DistributionHandle::~DistributionHandle()
{

}

DistributionHandle &DistributionHandle::operator=(const DistributionHandle &handle)
{
  DataBlockHandle::operator =(handle);
//   mBasis = handle.mBasis;
  return *this;
}

int DistributionHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);

  return 1;
}

int DistributionHandle::attachXMLInternal(XMLNode &node) const
{
  DataBlockHandle::attachXMLInternal(node);
  
  return 1;
}


}
