#include "BasisHandle.h"

namespace HDFileFormat{

double Basis::zero = 0.0;

const std::string BasisHandle::sDefaultBasisName = "Basis";

BasisHandle::BasisHandle(HandleType t)
  :DataBlockHandle(t)
{
  mID = sDefaultBasisName;  
}

BasisHandle::BasisHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename, t)
{
  mID = sDefaultBasisName;  
}

BasisHandle::BasisHandle(const BasisHandle &handle)
  :DataBlockHandle(handle),
   mBasis(handle.mBasis)
{
  mID = sDefaultBasisName;  
}

BasisHandle::~BasisHandle()
{
  //printf("<BasisHandle> destructor: <%s>\n", mID.c_str());  
}

BasisHandle &BasisHandle::operator=(const BasisHandle &handle)
{
  DataBlockHandle::operator =(handle);
  mBasis = handle.mBasis;
  return *this;
}

FileHandle &BasisHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_DATAPOINTS: //data points storing points
      return FileHandle::add(handle);
      break;
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside cluster.",handle.typeName());      
      break;
  }

  return *this;  
}  

void BasisHandle::setBasis(Basis &basis)
{
  mBasis = basis;
  
  //set the buffer
  DataBlockHandle::setData(&mBasis.coeffColMajor[0], mBasis.rows, mBasis.cols);
    
}

Basis& BasisHandle::basis()
{
  if(mBasis.empty())
  {
    //read the basis from the file
    //fprintf(stderr, "----Implement read basis from file----\n");
    mBasis.resize(mSampleCount, mDimension);
    if(this->isValid())
      readData(&mBasis.coeffColMajor[0]);
  }
  return mBasis;
}

int BasisHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);
  
  return 1;
}

int BasisHandle::attachXMLInternal(XMLNode &node) const
{
  DataBlockHandle::attachXMLInternal(node);

  return 1;    
}


}
