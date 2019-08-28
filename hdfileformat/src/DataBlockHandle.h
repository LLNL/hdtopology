#ifndef DATA_BLOCK_HANDLE_H
#define DATA_BLOCK_HANDLE_H

#include "FileHandle.h"
#include "HDFileFormatUtility.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>


/*
 * Provide the functionality to read and write data buffer into file
 * Can have DataBlockHandle children
 * (make the datablock as the universal node)
*/

namespace HDFileFormat{

class DataBlockHandle : public FileHandle
{
public:

  friend class DataCollectionHandle;

  friend class SegmentationHandle;

  //! The default name of the data set
  static const std::string sDefaultDataBlockName;

  //! Constructor
  explicit DataBlockHandle(HandleType t=H_DATABLOCK);

  //! Constructor
  DataBlockHandle(const char *filename, HandleType t=H_DATABLOCK);

  //! Copy constructor
  DataBlockHandle(const DataBlockHandle& handle);

  //! Destructor
  virtual ~DataBlockHandle();

  //! Assignment operator
  DataBlockHandle& operator=(const DataBlockHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new DataBlockHandle(*this);}

  //! instantiate -  load the mDataBuffer by reading from the file
  virtual void instantiate();

  //! instantiate -  load the mDataBuffer by reading from the file
  virtual void instantiateBuffer();

  //! Return the name of this handle
  std::string name() const {return mHandleName;}

  //! Set the name of the handle
  void name(std::string& n) {mHandleName = n;}

  //! Set the name of the handle
  void name(const char* n) {mHandleName = std::string(n);}


  //! Add a child but do a type-check
  //virtual FileHandle& add(const FileHandle& handle);

  //! Get the number of samples
  uint32_t sampleCount() const {return mSampleCount;}

  //! Get the dimension
  uint32_t dimension() const {return mDimension;}

  //! Get the size per value
  uint32_t valueSize() const {return mValueSize;}

  //! Get the name of the data type
  std::string dataType() const {return mDataType;}

  //! Get the file size
  uint32_t size() const {return mSampleCount*mDimension*mValueSize;}

  //! Set size
  void size(uint32_t bufferSize) {mSampleCount=bufferSize; mDimension=1; mValueSize=1;}

  //! Set the number of samples
  void sampleCount(uint32_t n) {mSampleCount = n;}

  //! Set the dimension of each sample
  void dimension(uint32_t d);

  //! Set the size in bytes per-value
  void valueSize(uint32_t s) {mValueSize = s;}

  //! Set the data type
  void dataType(const std::string& t) {mDataType = t;}

  //! Set the data (careful this does *not* copy the data)
  void setData(void *data);

  //! Single function to set the essential parameter for data writing
  template<class DataType>
  void setData(DataType* data, int pointCount, int dimension = 1);

  //! Read the data into the given array
  virtual void readData(void* data);

  //! Read the data into the given array
  template <class DataType>
  void readData(DataType* data);

  //! get compression flag
  bool compressionFlag(){return mCompressionFlag;}

  //! set compression flag
  void compressionFlag(bool flag){mCompressionFlag = flag;}

protected:
  //! A void pointer to the data used for writing
  void* mDataBuffer;

  //! A void pointer to the compressed data
  void* mDataBufferCompressed;

  //! compressed buffer size
  long long mCompressedBufferSize;

  //! Name of the handle
  std::string mHandleName;

  //! The number of samples
  uint32_t mSampleCount;

  //! The number of values for each sample
  uint32_t mDimension;

  //! The number of bytes per-value
  uint32_t mValueSize;

  //! The underlying datatype if known
  std::string mDataType;

  //! Compression flag, determine if the compression for the datablock is on or off
  bool mCompressionFlag;

  //! The attribute names
  //std::vector<std::string> mAttributeNames;

  //! Reset all values to their default uninitialized values
  void clear();

  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

  //! Write the data to the given file stream
  /*! If appropriate this function will write the corresponding in memory data
   *  to the given file stream. The filename and offsets will be adjusted
   *  relative to the new file. The handle effectively becomes linked to the new
   *  file.
   *  @param output: A filepointer to the current end of the new file
   *  @return 1 if successful; 0 otherwise.
   */
  virtual int writeDataInternal(std::ofstream& output, const std::string& filename);

protected:
  void doCompression();
  void doDeCompression();
};

template <class DataType>
inline void DataBlockHandle::readData(DataType* data)
{
  std::ifstream file;

  openInputFile(mFileName,file,!mASCIIFlag);
  rewind(file);

  if(this->mCompressionFlag)
  {
    if (this->mASCIIFlag) {
      hderror(true,"Cannot parse void data as ASCII");
    }
    else {
      //allocate mDataBufferCompressed
      if(!mDataBufferCompressed)
        mDataBufferCompressed = malloc(mCompressedBufferSize);
      file.read((char*)mDataBufferCompressed, mCompressedBufferSize);
    }
    //do the decompression
    doDeCompression();
    //copy to data, this will keep a copy of the uncompressed data in memory
    memcpy(data, mDataBuffer, size());
  }
  else
  {
    if (this->mASCIIFlag) {
      // for (uint32_t i=0;i<mSampleCount*mDimension;i++)
      //   file >> data[i];
    }
    else {
      file.read((char*)data,size());
    }
  }

  file.close();
}

template<class DataType>
inline void DataBlockHandle::setData(DataType *data, int sampleCount, int dimension)
{
  mDataBuffer = data;
  mSampleCount = sampleCount;
  mDimension = dimension;
  mValueSize = sizeof(DataType);

  mDataType = std::string( identifyTypeByPointer(data) );
}

}

#endif
