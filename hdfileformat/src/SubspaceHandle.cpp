#include "SubspaceHandle.h"

namespace HDFileFormat{

const std::string SubspaceHandle::sDefaultSubspaceName = "Subspace";

SubspaceHandle::SubspaceHandle(HDFileFormat::HandleType t)
  :FileHandle(t),
   mSubspaceNum(0)
{
  mID = sDefaultSubspaceName;
}

SubspaceHandle::SubspaceHandle(const char *filename, HandleType t)
  :FileHandle(filename,t),
   mSubspaceNum(0)
{
  mID = sDefaultSubspaceName;
  //subspace specific
}

SubspaceHandle::SubspaceHandle(const SubspaceHandle &handle)
  :FileHandle(handle),
   mSubspaceNum(handle.mSubspaceNum)
{
  //subspace specific
}

SubspaceHandle::~SubspaceHandle()
{
  //printf("{SubspaceHandle} destructor: {%s}\n", mID.c_str());
}

SubspaceHandle &SubspaceHandle::operator=(const SubspaceHandle &handle)
{
  FileHandle::operator=(handle);
  mSubspaceNum = handle.mSubspaceNum;

  return *this;
}

FileHandle &SubspaceHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_BASIS:
      return FileHandle::add(handle);
      break;
    case H_DATABLOCK:
    case H_SUBSPACE:
    case H_CLUSTER:
    case H_EMBEDDING:
    case H_COLLECTION:
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside datasets.",handle.typeName());
      return *this;
  }

  return *this;
}

void SubspaceHandle::setChildrenBasis(std::vector<HDFileFormat::Basis> &basis)
{
  //cleanup
  for(size_t i=0; i<mChildren.size(); i++)
    delete mChildren[i];
  mChildren.clear();

  mSubspaceNum = basis.size();
  for(size_t i=0; i<basis.size(); i++)
  {
    FileHandle *handle = this->constructHandle("Basis", mFileName);
    dynamic_cast<BasisHandle*>(handle)->setBasis(basis[i]);
    mChildren.push_back(handle);
  }
}

HDFileFormat::Basis &SubspaceHandle::subspaceBasisByIndex(int index)
{
  if(index>=0 && index<mChildren.size())
  {
    return dynamic_cast<BasisHandle*>(mChildren[index])->basis();
  }
  else
    return dynamic_cast<BasisHandle*>(mChildren[0])->basis();
}

int SubspaceHandle::parseXMLInternal(const XMLNode &node)
{
  FileHandle::parseXMLInternal(node);

  getAttribute(node,"SubspaceSize", mSubspaceNum);

  return 1;
}

int SubspaceHandle::attachXMLInternal(XMLNode &node) const
{
  FileHandle::attachXMLInternal(node);

  addAttribute(node,"SubspaceSize", mSubspaceNum);

  return 1;
}


}//namespace
