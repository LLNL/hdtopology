#ifndef POINTSET_META_INFO_H
#define POINTSET_META_INFO_H

/*
Object for storing per-point meta info

One PointSetHandle can have multiple PointSetMetaInfoHandle

PointSetMetaInfo hold data read from PointSetMetaInfoHandle(handle don't hold data)

*/
#include "HDFileFormatUtility.h"
#include <string>
#include <vector>
#include <stdlib.h>

namespace HDFileFormat{

enum MetaInfoType
{
  META_INFO_STRING,
  META_INFO_IMAGE,
  META_INFO_UNDFEINED
};

//library independent image presentation
//uncompressed, no color table, always 8bit for displaying
struct imageData
{
  int width, height, channels;
  std::vector<unsigned char> buffer;
};

class PointSetMetaInfo
{
  //allow PointSetMetaInfoHandle to access member directly
  friend class DataPointsMetaInfoHandle;

public:
  PointSetMetaInfo();

  MetaInfoType GetMetaInfoType(){return m_metaInfoType;}

  //set
  template<typename DataType>
  void SetImageData(std::vector<std::vector<DataType> > &imageList, int imageX, int imageY, int channelCount=1);
  void SetStringData(std::vector<std::string> &stringList);

  //get
  template<typename DataType>
  void GetImageData(std::vector<std::vector<DataType> > &imageList, int &imageX, int &imageY, int &channelCount);
  template<typename DataType>
  void GetImageData(std::vector<std::vector<DataType> > &imageList);
  int GetImageX(){return m_imageSizeX;}
  int GetImageY(){return m_imageSizeY;}
  int GetChannelNum(){return m_channelCount;}
  std::string GetDataTypeString(){return m_dataType;}

  //get string metaInfo
  void GetStringData(std::vector<std::string> &stringList);

protected:
  //for string
  int m_stringLength;

  //for image
  int m_imageSizeX, m_imageSizeY, m_channelCount;

  MetaInfoType m_metaInfoType;

  void* m_data;
  size_t m_dataByteSize;
  size_t m_dataTypeSize;

  std::string m_dataType;
};

template<typename DataType>
inline void PointSetMetaInfo::SetImageData(std::vector<std::vector<DataType> > &imageList, int imageX, int imageY, int channelCount)
{
  m_metaInfoType = META_INFO_IMAGE;

  if(imageList.size() == 0)
    return;

  if(m_data)
  {
    delete (char*)m_data;
  }

  m_imageSizeX = imageX;
  m_imageSizeY = imageY;
  m_channelCount = channelCount;
  m_dataTypeSize = sizeof(DataType);

  DataType* dummy = NULL;
  m_dataType = std::string( identifyTypeByPointer(dummy) );
  m_dataByteSize = imageX*imageY*channelCount*sizeof(DataType)*imageList.size();

  //convert to m_data buffer
  m_data = malloc(m_dataByteSize);
  for(size_t i=0; i<imageList.size(); i++)
    for(int j=0; j<imageX*imageY*channelCount; j++)
      ((DataType*)m_data)[i*imageX*imageY*channelCount + j] = imageList[i][j];
}

template<typename DataType>
inline void PointSetMetaInfo::GetImageData(std::vector<std::vector<DataType> > &imageList, int &imageX, int &imageY, int &channelCount)
{
  GetImageData(imageList);
  imageX = m_imageSizeX;
  imageY = m_imageSizeY;
  channelCount = m_channelCount;
}

template<typename DataType>
inline void PointSetMetaInfo::GetImageData(std::vector<std::vector<DataType> > &imageList)
{
  if(m_metaInfoType == META_INFO_IMAGE)
  {
    imageList.clear();
    DataType* dummy = NULL;
    if(m_dataType == std::string(identifyTypeByPointer(dummy)) )
    {
      int numberOfImage = m_dataByteSize/(m_imageSizeX*m_imageSizeY*m_channelCount*sizeof(DataType));
      imageList.resize(numberOfImage);
      for(int i=0; i<numberOfImage; i++)
      {
        for(int j=0; j<m_imageSizeX*m_imageSizeY*m_channelCount; j++)
          imageList[i].push_back( ((DataType*)m_data)[ i*m_imageSizeX*m_imageSizeY*m_channelCount + j ] );
      }
    }
  }
}




}//namespace

#endif
