//single file compression library (similar to zlib)
#include <external/miniz.c>

#include "DataBlockHandle.h"

namespace HDFileFormat{

const std::string DataBlockHandle::sDefaultDataBlockName = "Generic DataBlock";

DataBlockHandle::DataBlockHandle(HandleType t)
  :FileHandle(t),
   mDataBuffer(NULL),
   mHandleName(sDefaultDataBlockName),
   mSampleCount(0),
   mDimension(0),
   mValueSize(0),
   mDataType(""),
   mCompressionFlag(false),
   mCompressedBufferSize(0),
   mDataBufferCompressed(NULL)
{
  mID = sDefaultDataBlockName;
}

DataBlockHandle::DataBlockHandle(const char *filename, HandleType t)
  :FileHandle(filename,t),
   mDataBuffer(NULL),
   mHandleName(sDefaultDataBlockName),
   mSampleCount(0),
   mDimension(0),
   mValueSize(0),
   mDataType(""),
   mCompressionFlag(false),
   mCompressedBufferSize(0),
   mDataBufferCompressed(NULL)
{
  mID = sDefaultDataBlockName;
}

DataBlockHandle::DataBlockHandle(const DataBlockHandle& handle)
  :FileHandle(handle),
   mDataBuffer(handle.mDataBuffer),
   mSampleCount(handle.mSampleCount),
   mDimension(handle.mDimension),
   mHandleName(handle.mHandleName),
   mValueSize(handle.mValueSize),
   mDataType(handle.mDataType),
   mCompressionFlag(handle.mCompressionFlag),
   mCompressedBufferSize(handle.mCompressedBufferSize),
   mDataBufferCompressed(handle.mDataBufferCompressed)
{
}

DataBlockHandle::~DataBlockHandle()
{
}

DataBlockHandle& DataBlockHandle::operator=(const DataBlockHandle& handle)
{
  FileHandle::operator=(handle);

  mDataBuffer = handle.mDataBuffer;
  mHandleName = handle.mHandleName;
  mSampleCount = handle.mSampleCount;
  mDimension = handle.mDimension;
  mValueSize = handle.mValueSize;
  mDataType = handle.mDataType;
  mCompressionFlag = handle.mCompressionFlag;
  mCompressedBufferSize = handle.mCompressedBufferSize;
  mDataBufferCompressed = handle.mDataBufferCompressed;
  
  return *this;
}

void DataBlockHandle::instantiate()
{
  this->instantiateBuffer();

  FileHandle::instantiate();
}

void DataBlockHandle::instantiateBuffer()
{
  mDataBuffer = malloc(size());
  this->readData(mDataBuffer);
}

//FileHandle& DataBlockHandle::add(const FileHandle& handle)
//{
//  switch (handle.type()) {
//    case H_COLLECTION:
//    case H_DATASET:
//    case H_DATABLOCK:
//    default:
//      hderror(true,"Nodes of type \"%s\" cannot be nested inside datasets.",handle.typeName());
//      return *this;
//  }

//  return *this;
//}

void DataBlockHandle::dimension(uint32_t d)
{
  mDimension = d;
}

void DataBlockHandle::setData(void *data)
{
  mDataBuffer = data;
}


void DataBlockHandle::readData(void* data)
{
  std::ifstream file;

  openInputFile(mFileName,file,!mASCIIFlag);
  rewind(file);

  //if need decompression
  if(mCompressionFlag)
  {
    if (this->mASCIIFlag) {
      hderror(true,"Cannot parse void data as ASCII");
    }
    else {
      //allocate mDataBufferCompressed
      if(!mDataBufferCompressed)
        mDataBufferCompressed = malloc(mCompressedBufferSize);
      file.read((char*)mDataBufferCompressed, mCompressedBufferSize);
      //do the decompression
      doDeCompression();
      //copy to data, this will keep a copy of the uncompressed data in memory
      memcpy(data, mDataBuffer, size());
    }
  }
  else
  {
    if (this->mASCIIFlag) {
      hderror(true,"Cannot parse void data as ASCII");
    }
    else {
      file.read((char*)data,size());
    }
  }

  file.close();
}

void DataBlockHandle::clear()
{
  FileHandle::clear();

  mDataBuffer = NULL;
  mSampleCount = 0;
  mDimension = 0;
  mValueSize = 0;
  mDataType = "";
  //mAttributeNames.clear();
}

int DataBlockHandle::parseXMLInternal(const XMLNode& node)
{
  FileHandle::parseXMLInternal(node);

  getAttribute(node,"samplecount",mSampleCount);
  getAttribute(node,"dimension",mDimension);
  getAttribute(node,"valuesize",mValueSize);
  getAttribute(node,"datatype",mDataType);

  getAttribute(node, "compression", mCompressionFlag);
  getAttribute(node, "compressionBufferLength", mCompressedBufferSize);

  return 1;
}

int DataBlockHandle::attachXMLInternal(XMLNode& node) const
{
  FileHandle::attachXMLInternal(node);

  addAttribute(node,"samplecount",mSampleCount);
  addAttribute(node,"dimension",mDimension);
  addAttribute(node,"valuesize",mValueSize);
  addAttribute(node,"datatype",mDataType.c_str());

  addAttribute(node, "compression", mCompressionFlag);
  addAttribute(node, "compressionBufferLength", mCompressedBufferSize);

  return 1;
}

int DataBlockHandle::writeDataInternal(std::ofstream& output, const std::string& filename)
{
  this->mFileName = filename;

  //making progressive write possible
  if(this->mOffset == -1) //if not inited
    this->mOffset = output.tellp();
  else
    output.seekp(this->mOffset);

  /////////////// if compression is ON ///////////////////////////////////////
  if(mCompressionFlag)
  {
    //compress
    if(mCompressionFlag)
      doCompression();
    //if there is data to write
    if (mDataBufferCompressed != NULL)
    {
      if (mASCIIFlag) {
        hderror(true,"Cannot parse void data as ASCII");
      }
      else {
        output.write((const char*)mDataBufferCompressed, mCompressedBufferSize);
      }
    }
    else
    {
      //if don't write seek to the end of the write
      output.seekp(this->mOffset+this->mCompressedBufferSize);
    }
  }
  else ////////////// if compression is off ////////////////////////////////////
  {
    //if there is data to write
    if (mDataBuffer != NULL)
    {
      if (mASCIIFlag) {
        hderror(true,"Cannot parse void data as ASCII");
        //for (uint32_t i=0;i<mDimension*mSampleCount;i++)
        //  output << mDataBuffer[i] << "\n";
      }
      else {
        output.write((const char*)mDataBuffer,size());
      }
    }
    else
    {
      //if don't write seek to the end of the write
      output.seekp(this->mOffset+this->size());
    }
  }

  return 1;
}

void DataBlockHandle::doCompression()
{
  unsigned long destLength = compressBound(size());
  mz_uint8 *destBuffer = (mz_uint8 *)malloc((size_t)destLength);
  int status = compress( (Byte*)(destBuffer), &destLength, (const Byte *)(mDataBuffer), uLong(size()) );

  if(status == Z_OK)
  {
    //copy to buffer with correct length
    this->mCompressedBufferSize = destLength;
    mDataBufferCompressed = malloc(mCompressedBufferSize);
    memcpy(this->mDataBufferCompressed, destBuffer, destLength);
  }
  else
  {
    hderror(this, "Compression failed ...");
  }

  //cleanup
  free(destBuffer);
}

void DataBlockHandle::doDeCompression()
{
  mDataBuffer = malloc(size());
  mz_ulong uncompressedLength = size();
  int status = uncompress((unsigned char*)mDataBuffer, &uncompressedLength,
    (const unsigned char*)mDataBufferCompressed, mz_ulong(mCompressedBufferSize) );

  if(status != Z_OK)
  {
    hderror(this, "Decompression failed ...");
  }
}


} //namespace
