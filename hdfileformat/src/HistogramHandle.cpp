#include "HistogramHandle.h"

namespace HDFileFormat{

const std::string HistogramHandle::sDefaultBasisName = "Histogram";

HistogramHandle::HistogramHandle(HandleType t)
  :DataBlockHandle(t)
{
  mID = sDefaultBasisName;
}

HistogramHandle::HistogramHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename, t)
{
  mID = sDefaultBasisName;
}

HistogramHandle::HistogramHandle(const HistogramHandle &handle)
  :DataBlockHandle(handle)
//    mBasis(handle.mBasis)
{
  mID = sDefaultBasisName;
}

HistogramHandle::~HistogramHandle()
{

}

HistogramHandle &HistogramHandle::operator=(const HistogramHandle &handle)
{
  DataBlockHandle::operator =(handle);
//   mBasis = handle.mBasis;
  return *this;
}

FileHandle &HistogramHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_DATABLOCK:
      return FileHandle::add(handle);
      break;
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside cluster.",handle.typeName());
      break;
  }

  return *this;
}

int HistogramHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);

  return 1;
}

int HistogramHandle::attachXMLInternal(XMLNode &node) const
{
  DataBlockHandle::attachXMLInternal(node);

  return 1;
}


}
