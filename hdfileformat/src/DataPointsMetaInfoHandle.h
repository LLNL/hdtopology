#ifndef DATA_POINTS_META_INFO_HANDLE_H
#define DATA_POINTS_META_INFO_HANDLE_H

#include "DataBlockHandle.h"
#include "PointSetMetaInfo.h"

/*

DataPointsMetaInfo store the non-numerical and irregular per-point data such as per-point label, image, and other meta information

Author: Shusen Liu
Date: Feb 27, 2015

String can be store at array with fix length(the maximum string length in the data, or preset length like 50)

*/

namespace HDFileFormat{

class DataPointsMetaInfoHandle: public DataBlockHandle
{
public:
  //! The default name of the data set
  static const std::string sDefaultDataPointsMetaInfoName;

  //! Constructor
  explicit DataPointsMetaInfoHandle(HandleType t=H_DATAPOINTS_METAINFO);

  //! Constructor
  DataPointsMetaInfoHandle(const char *filename, HandleType t=H_DATAPOINTS_METAINFO);

  //! Copy constructor
  DataPointsMetaInfoHandle(const DataPointsMetaInfoHandle& handle);

  //! Destructor
  virtual ~DataPointsMetaInfoHandle();

  //! Assignment operator
  DataPointsMetaInfoHandle &operator=(const DataPointsMetaInfoHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new DataPointsMetaInfoHandle(*this);}

  //! Set MetaInfo
  void SetMetaInfo(PointSetMetaInfo &metaInfo);

  //! Read MetaInfo
  void ReadMetaInfo(PointSetMetaInfo &metaInfo);

protected:
  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:
  //for string
  int mStringLength;

  //for image
  int mImageWidth, mImageHeight, mChannelCount;

  //type
  MetaInfoType mMetaInfoType;

};

}//namespace

#endif
