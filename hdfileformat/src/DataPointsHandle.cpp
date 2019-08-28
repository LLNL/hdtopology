#include "DataPointsHandle.h"
#include <iostream>

namespace HDFileFormat{

const std::string DataPointsHandle::sDefaultDataPointsName = "HDDataPoints";

DataPointsHandle::DataPointsHandle(HandleType t)
  :DataBlockHandle(t)
{

}

DataPointsHandle::DataPointsHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename,t)
{

}

DataPointsHandle::DataPointsHandle(const DataPointsHandle& handle)
  :DataBlockHandle(handle),
   mAttributeNames(handle.mAttributeNames),
   mSpatialDim(handle.mSpatialDim),
   mDimensionFlag(handle.mDimensionFlag)
{
}

DataPointsHandle::~DataPointsHandle()
{
}

DataPointsHandle& DataPointsHandle::operator=(const DataPointsHandle& handle)
{
  DataBlockHandle::operator=(handle);

  mAttributeNames = handle.mAttributeNames;
  mSpatialDim = handle.mSpatialDim;
  mDimensionFlag = handle.mDimensionFlag;

  return *this;
}

FileHandle& DataPointsHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_DATAPOINTS_METAINFO:
      return FileHandle::add(handle);
      //printf("Points children: %ld\n",this->mChildren.size());
      break;
    case H_GRAPH:
      return FileHandle::add(handle);
      break;
    case H_DATAPOINTS:
      return FileHandle::add(handle);
      break;
    case H_SEGMENTATION:
      return FileHandle::add(handle);
      break;
    case H_EMBEDDING:
      return FileHandle::add(handle);
      break;
    case H_FUNCTION:
      return FileHandle::add(handle);
      break;
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside datasets.",handle.typeName());
      break;
  }

  // YOu should never reach this
  return *this;
}

void DataPointsHandle::setDimensionLabel(std::vector<std::string> &dimLabel)
{
  if (mAttributeNames.size() != mDimension)
      mAttributeNames.resize(mDimension);

  //auto setup the default values
  if(dimLabel.size() != mDimension && mDimension>0)
  {
    for (uint32_t i=0;i<mDimension;i++)
    {
      std::stringstream token;
      token << "Attribute " << i;

      mAttributeNames[i] = token.str();

    }
  }
  else
  {
    for(size_t i=0; i<dimLabel.size(); i++)
      mAttributeNames[i] = dimLabel[i];
  }
}

void DataPointsHandle::setDimensionFlag(std::vector<bool> &dimFlag)
{
  if(mDimensionFlag.size() != mDimension)
    mDimensionFlag.resize(mDimension);

  //set default
  if(dimFlag.size() != mDimension && mDimension>0)
  {
    for (uint32_t i=0;i<mDimension;i++)
    {
      //set default flag
      mDimensionFlag[i] = true;
    }
  }
  else
    for(size_t i=0; i<dimFlag.size(); i++)
      mDimensionFlag[i] = dimFlag[i];
}

void DataPointsHandle::setSpatialDim(uint32_t dimX, uint32_t dimY, uint32_t dimZ)
{
  mSpatialDim.clear();

  if(dimX)
    mSpatialDim.push_back(dimX);
  else
    return;

  if(dimY)
    mSpatialDim.push_back(dimY);
  else
    return;

  if(dimZ)
    mSpatialDim.push_back(dimZ);
  else
    return;

}


void DataPointsHandle::getSpatialDim(uint32_t &dimX, uint32_t &dimY, uint32_t &dimZ)
{
  uint32_t dim[3];

  for(size_t i=0; i<3; i++)
    if(i >= mSpatialDim.size())
      dim[i] = 1;
    else
      dim[i] = mSpatialDim[i];

  dimX = dim[0];
  dimY = dim[1];
  dimZ = dim[2];
}

void DataPointsHandle::getSpatialDim(int &dimX, int &dimY, int &dimZ)
{
  int dim[3];

  for(size_t i=0; i<3; i++)
    if(i >= mSpatialDim.size())
      dim[i] = 1;
    else
      dim[i] = mSpatialDim[i];

  dimX = dim[0];
  dimY = dim[1];
  dimZ = dim[2];
}

void DataPointsHandle::clear()
{
  DataBlockHandle::clear();
  mAttributeNames.clear();
}

int DataPointsHandle::parseXMLInternal(const XMLNode& node)
{
  DataBlockHandle::parseXMLInternal(node);

  //read attribute names
  std::string names;
  getAttribute(node,"attributeNames",names);
  mAttributeNames = splitString(names);

  //read attribute flag
  std::string flags;
  getAttribute(node,"dimensionFlag",flags);
  if(flags.empty())
    mDimensionFlag = splitStringToNumber<bool>(flags);

  //read spaital dimension
  int spatialDim;
  getAttribute(node, "spatialDim", spatialDim);
  mSpatialDim.resize(spatialDim);

  if(mSpatialDim.size() == 2)
  {
    getAttribute(node, "dimX", mSpatialDim[0]);
    getAttribute(node, "dimY", mSpatialDim[1]);
  }else if(mSpatialDim.size() == 3)
  {
    getAttribute(node, "dimX", mSpatialDim[0]);
    getAttribute(node, "dimY", mSpatialDim[1]);
    getAttribute(node, "dimZ", mSpatialDim[2]);
  }

  return 1;
}

int DataPointsHandle::attachXMLInternal(XMLNode& node) const
{
  DataBlockHandle::attachXMLInternal(node);

  std::stringstream output;
  for (uint32_t i=0;i<mAttributeNames.size();i++) {
    if (i > 0)
      output << sStringSeperator;
    output << mAttributeNames[i];
  }

  addAttribute(node,"attributeNames",output.str());

  output.str(std::string());
  output.clear();
  for (uint32_t i=0;i<mDimensionFlag.size();i++) {
    if (i > 0)
      output << sStringSeperator;
    output << mDimensionFlag[i];
  }
  //only write dimen flag if mDimensionFlag is set
  if(output.str().empty())
    addAttribute(node, "dimensionFlag", output.str());

  addAttribute(node, "spatialDim", mSpatialDim.size());

  if(mSpatialDim.size() == 2)
  {
    addAttribute(node, "dimX", mSpatialDim[0]);
    addAttribute(node, "dimY", mSpatialDim[1]);
  }else if(mSpatialDim.size() == 3)
  {
    addAttribute(node, "dimX", mSpatialDim[0]);
    addAttribute(node, "dimY", mSpatialDim[1]);
    addAttribute(node, "dimZ", mSpatialDim[2]);
  }

  //std::cout<<std::string(output.str())<<std::endl;
  return 1;
}


}
