#ifndef DATA_POINTS_HANDLE_H
#define DATA_POINTS_HANDLE_H

#include "DataBlockHandle.h"

namespace HDFileFormat{

class DataPointsHandle: public DataBlockHandle
{
public:

  //! The default name of the data set
  static const std::string sDefaultDataPointsName;

  //! Constructor
  explicit DataPointsHandle(HandleType t=H_DATAPOINTS);

  //! Constructor
  DataPointsHandle(const char *filename, HandleType t=H_DATAPOINTS);

  //! Copy constructor
  DataPointsHandle(const DataPointsHandle& handle);

  //! Destructor
  virtual ~DataPointsHandle();

  //! Assignment operator
  DataPointsHandle& operator=(const DataPointsHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new DataPointsHandle(*this);}

  //! Add a child but do a type-check
  virtual FileHandle& add(const FileHandle& handle);

  //! Set Dimension, with default dimension label
  void setDimensionLabel(std::vector<std::string> &dimLabel);

  //! Set Dimension flag, decided if the corresponding dimension is active or not
  void setDimensionFlag(std::vector<bool> &dimFlag);

  //! Add dimension to the dataPoints
  template<typename dataType>
  void addDimension(dataType *buffer, int size, std::string dimLabel);

  //! Get the dimension flags
  const std::vector<bool> dimensionFlags(){return mDimensionFlag;}

  //! Get the attribute names
  const std::vector<std::string>& attributeNames() const {return mAttributeNames;}

  //! Get the i'th attribute name
  std::string attributeName(uint32_t i) const {return mAttributeNames[i];}

  //! Set the attribute names
  void attributeNames(const std::vector<std::string>& names) {mAttributeNames = names;}

  //! Set the i'th attribute name
  void attributeName(uint32_t i,const std::string& name) {mAttributeNames[i] = name;}

  //! Set spatial dimensions
  void setSpatialDim(uint32_t dimX, uint32_t dimY, uint32_t dimZ);

  //! Get spatial dimensions
  void getSpatialDim(uint32_t &dimX, uint32_t &dimY, uint32_t &dimZ);
  void getSpatialDim(int &dimX, int &dimY, int &dimZ);

  //! query spatial dimension
  uint32_t spatialDim(){return uint32_t(mSpatialDim.size());}

protected:

  //! The attribute names
  std::vector<std::string> mAttributeNames;

  //! The flag indicating the corresponding dimension is active or not
  std::vector<bool> mDimensionFlag;

  //! volume spatial dimension for volume implicit coordinate, default to (pointCount,0,0)
  std::vector<uint32_t> mSpatialDim;

  //! Reset all values to their default uninitialized values
  void clear();

  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;
};

template<class DataType>
void DataPointsHandle::addDimension(DataType *buffer, int size, std::string dimLabel)
{
  //FIXME
}



} //namespace


#endif
