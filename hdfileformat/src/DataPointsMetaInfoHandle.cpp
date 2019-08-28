#include "DataPointsMetaInfoHandle.h"

namespace HDFileFormat{

const std::string DataPointsMetaInfoHandle::sDefaultDataPointsMetaInfoName = "HDDataMetaInfo";

DataPointsMetaInfoHandle::DataPointsMetaInfoHandle(HandleType t)
  :DataBlockHandle(t),
   mMetaInfoType(META_INFO_UNDFEINED)
{

}

DataPointsMetaInfoHandle::DataPointsMetaInfoHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename, t),
   mMetaInfoType(META_INFO_UNDFEINED)
{

}

DataPointsMetaInfoHandle::DataPointsMetaInfoHandle(const DataPointsMetaInfoHandle &handle)
  :DataBlockHandle(handle),
   mStringLength(handle.mStringLength),
   mImageWidth(handle.mImageWidth),
   mImageHeight(handle.mImageHeight),
   mChannelCount(handle.mChannelCount),
   mMetaInfoType(handle.mMetaInfoType)
  //other class members
{

}

DataPointsMetaInfoHandle::~DataPointsMetaInfoHandle()
{

}

DataPointsMetaInfoHandle &DataPointsMetaInfoHandle::operator=(const DataPointsMetaInfoHandle &handle)
{
  DataBlockHandle::operator=(handle);

  //assign other members in the class
  mStringLength = handle.mStringLength;
  mImageWidth = handle.mImageWidth;
  mImageHeight = handle.mImageHeight;
  mChannelCount = handle.mChannelCount;
  mMetaInfoType = handle.mMetaInfoType;

  return *this;
}

void DataPointsMetaInfoHandle::SetMetaInfo(PointSetMetaInfo &metaInfo)
{
  mDataBuffer = metaInfo.m_data;
  mSampleCount = metaInfo.m_dataByteSize/metaInfo.m_dataTypeSize;
  mDimension = 1;
  mValueSize = metaInfo.m_dataTypeSize;
  mDataType = metaInfo.m_dataType;

  mMetaInfoType = metaInfo.m_metaInfoType;
  if(mMetaInfoType == META_INFO_STRING)
  {
    mStringLength = metaInfo.m_stringLength;
  }
  else if(mMetaInfoType == META_INFO_IMAGE)
  {
    //printf("set image metaInfo\n");
    mImageWidth = metaInfo.m_imageSizeX;
    mImageHeight = metaInfo.m_imageSizeY;
    mChannelCount = metaInfo.m_channelCount;
  }
}

void DataPointsMetaInfoHandle::ReadMetaInfo(PointSetMetaInfo &metaInfo)
{
  metaInfo.m_dataTypeSize = mValueSize;
  metaInfo.m_dataByteSize = mSampleCount*mValueSize;
  metaInfo.m_dataType = mDataType;

  metaInfo.m_data = malloc(metaInfo.m_dataByteSize);
  readData(metaInfo.m_data);

  metaInfo.m_metaInfoType = mMetaInfoType;
  if(mMetaInfoType == META_INFO_STRING)
  {
    metaInfo.m_stringLength = mStringLength;
  }
  else if(mMetaInfoType == META_INFO_IMAGE)
  {
    metaInfo.m_imageSizeX = mImageWidth;
    metaInfo.m_imageSizeY = mImageHeight;
    metaInfo.m_channelCount = mChannelCount;
  }
}


//////////////////////////////////////// protected //////////////////////////////////////

////////////////////////
/// \brief PointSetMetaInfo::parseXMLInternal
/// \param node
/// \return
///

int DataPointsMetaInfoHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);

  std::string typeString;
  getAttribute(node, "metaInfoType", typeString);

  if(typeString == std::string("string"))
  {
    mMetaInfoType = META_INFO_STRING;
    getAttribute(node, "stringLength", this->mStringLength);
  }
  else if(typeString == std::string("image"))
  {
    mMetaInfoType = META_INFO_IMAGE;
    getAttribute(node, "imageWidth", this->mImageWidth);
    getAttribute(node, "imageHeight", this->mImageHeight);
    getAttribute(node, "channelCount", this->mChannelCount);
  }
  return 1;
}

/////
/// \brief PointSetMetaInfo::attachXMLInternal
/// \param node
/// \return
///

int DataPointsMetaInfoHandle::attachXMLInternal(XMLNode &node) const
{
  DataBlockHandle::attachXMLInternal(node);

  if(mMetaInfoType == META_INFO_STRING)
  {
    addAttribute(node, "metaInfoType", "string");
    addAttribute(node, "stringLength", this->mStringLength);
  }
  else if(mMetaInfoType == META_INFO_IMAGE)
  {
    addAttribute(node, "metaInfoType", "image");
    addAttribute(node, "imageWidth", this->mImageWidth);
    addAttribute(node, "imageHeight", this->mImageHeight);
    addAttribute(node, "channelCount", this->mChannelCount);
  }
  return 1;
}

} //namespace
