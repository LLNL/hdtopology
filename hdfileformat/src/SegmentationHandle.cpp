/*
 * SegmentationHandle.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: bremer5
 */

#include <cassert>
#include "SegmentationHandle.h"
#include "FileData.h"

namespace HDFileFormat {

SegmentationHandle::SegmentationHandle() : FileHandle(H_SEGMENTATION),
    mSegCount(0), mSegSize(0), mHasIndexMap(false), mSegmentation(NULL), mSteepest(NULL),
    mFlatSegmentation(NULL), mOffsets(NULL), mIndexMap(0)
{
}

SegmentationHandle::SegmentationHandle(const char* filename) : FileHandle(filename,H_SEGMENTATION),
    mSegCount(0), mSegSize(0),mHasIndexMap(false),mSegmentation(NULL), mSteepest(NULL),
    mFlatSegmentation(NULL), mOffsets(NULL), mIndexMap(0)
{
}

SegmentationHandle::SegmentationHandle(const SegmentationHandle& handle) : FileHandle(handle)
{
  mSegCount = handle.mSegCount;
  mSegSize = handle.mSegSize;
  mHasIndexMap = handle.mHasIndexMap;
  mSegmentation = handle.mSegmentation;
  mFlatSegmentation = handle.mFlatSegmentation;
  mSteepest = handle.mSteepest;
  mOffsets = handle.mOffsets;
  mIndexMap = handle.mIndexMap;
}

SegmentationHandle::~SegmentationHandle()
{
}

SegmentationHandle& SegmentationHandle::operator=(const SegmentationHandle& handle)
{
  FileHandle::operator=(handle);

  mSegCount = handle.mSegCount;
  mSegSize = handle.mSegSize;
  mHasIndexMap = handle.mHasIndexMap;
  mSegmentation = handle.mSegmentation;
  mFlatSegmentation = handle.mFlatSegmentation;
  mSteepest = handle.mSteepest;
  mOffsets = handle.mOffsets;
  mIndexMap = handle.mIndexMap;

  return *this;
}

std::string SegmentationHandle::segmentationType() const
{
  std::string type("Segmentation");

  for (uint32_t i=0;i<this->mChildren.size();i++) {
    if (mChildren[i]->idString() == "Hierarchy")
      type = "HierarchicalSegmentation";
  }

  // If we couldn't find a hierarchy
  if (type == "Segmentation")
    return type;

  // Otherwise we look for saddle pairs
  for (uint32_t i=0;i<this->mChildren.size();i++) {
    if (mChildren[i]->idString() == "SaddlePairs")
      type = "MorseComplex";
  }

  return type;
}


void SegmentationHandle::setSegmentation(std::vector<std::vector<uint32_t> >* segmentation)
{
  mSegmentation = segmentation;
  mSegCount = mSegmentation->size();
}

void SegmentationHandle::setOffsets(std::vector<uint32_t>* offsets)
{
  mOffsets = offsets;
  mSegCount = mOffsets->size()-1;
}

void SegmentationHandle::setSegmentation(std::vector<uint32_t>* segmentation)
{
  mFlatSegmentation = segmentation;
}

void SegmentationHandle::setSteepest(std::vector<uint32_t> *steepest)
{
  mSteepest = steepest;
}

void SegmentationHandle::setIndexMap(std::vector<uint32_t>& index_map)
{
  mHasIndexMap = true;
  mIndexMap = index_map;
}

FileHandle& SegmentationHandle::add(const FileHandle& handle)
{
  switch (handle.type()) {
    case H_DATABLOCK:
      return FileHandle::add(handle);
      //this->mChildren.push_back(dynamic_cast<const DataBlockHandle&>(handle));
      //fprintf(stderr,"SegmentationHandle::add %d %d\n",mChildren.size(),childCount());
      //return this->mChildren.back();
      break;
    default:
      fprintf(stderr,"Unknown handle type cannot attach a %d to a segmentation",handle.type());
      assert(false);
      break;
  }

  // Note that this is a bogus return to satisfy the compiler. You should hit
  // the error before coming to here
  return *mChildren[0];
}


#ifdef ENABLE_STEEPEST
int SegmentationHandle::readSegmentation(std::vector<uint32_t>& offsets, std::vector<uint32_t>& segmentation,
                                         std::vector<uint32_t>& index_map, std::vector<uint32_t>& steepest)
#else
int SegmentationHandle::readSegmentation(std::vector<uint32_t>& offsets, std::vector<uint32_t>& segmentation,
                                         std::vector<uint32_t>& index_map)
#endif
{
  std::ifstream file;
  uint32_t size;

  openInputFile(mFileName,file,!mASCIIFlag);
  rewind(file);

  // Read offsets
  offsets.resize(mSegCount+1);
  Data<uint32_t> off(&offsets);

  if (mASCIIFlag)
    off.readASCII(file);
  else
    off.readBinary(file);

  // Read segmentation
  segmentation.resize(offsets.back());
  Data<uint32_t> seg(&segmentation);

  if (mASCIIFlag)
    seg.readASCII(file);
  else
    seg.readBinary(file);

#ifdef ENABLE_STEEPEST
  // Read steepest
  steepest.resize(offsets.back());
  Data<uint32_t> steepest(&steepest);
  if (mASCIIFlag)
    steepest.readASCII(file);
  else
    steepest.readBinary(file);
#endif

  // Read the rest of the data
  if (mHasIndexMap) {

    index_map.resize(mSegCount);

    Data<uint32_t> ind(&index_map);

    // Read the rest of the data
    if (mASCIIFlag)
      ind.readASCII(file);
    else
      ind.readBinary(file);
  }

  file.close();

  return 1;
}



/*
void SegmentationHandle::topHandle(FileHandle* top)
{
  this->mTopHandle = top;

  std::vector<DataBlockHandle>::iterator it;
  for (it=mChildren.begin();it!=mChildren.end();it++)
    it->topHandle(top);
}
*/

int SegmentationHandle::parseXMLInternal(const XMLNode& node)
{
  FileHandle::parseXMLInternal(node);

  if (node.getAttribute("segcount",0) == NULL) {
    fprintf(stderr,"Warning: no \"segcount\" attribute found in segmentation handle.");
    mSegCount = 0;
  }
  else {
    std::stringstream input(std::string(node.getAttribute("segcount",0)));
    input >> mSegCount;
  }

  if (node.getAttribute("hasIndexMap",0) == NULL) {
    fprintf(stderr,"Warning: no \"hasIndexMap\" attribute found in segmentation handle.");
    mHasIndexMap = false;
  }
  else {
    std::stringstream input(std::string(node.getAttribute("hasIndexMap",0)));
    input >> mHasIndexMap;
  }
  return 1;
}

int SegmentationHandle::attachXMLInternal(XMLNode& node) const
{
  XMLNode child;
  FileHandle::attachXMLInternal(node);

  addAttribute(node,"segcount",mSegCount);
  addAttribute(node,"hasIndexMap",mHasIndexMap);

  return 1;
}

int SegmentationHandle::writeDataInternal(std::ofstream& output, const std::string& filename)
{
  uint32_t count = 0;
  uint32_t i;
  std::vector<uint32_t>::iterator it;

  this->mFileName = filename;
  this->mOffset = static_cast<FileOffsetType>(output.tellp());

  //! If there is no data to write
  if (mSegmentation != NULL) {

    if (mASCIIFlag) {
      for (i=0;i<mSegmentation->size();i++) {

        output << count << std::endl;
        count += (*mSegmentation)[i].size();
      }
      output << count << std::endl;

      for (i=0;i<mSegmentation->size();i++) {
        for (it=(*mSegmentation)[i].begin();it!=(*mSegmentation)[i].end();it++) {
          output << *it << " ";
        }
        output << std::endl;
      }
    }
    else {
      for (i=0;i<mSegmentation->size();i++) {

        output.write((const char*)&count,sizeof(uint32_t));
        count += (*mSegmentation)[i].size();
      }

      for (i=0;i<mSegmentation->size();i++)
        output.write((const char*)&(*mSegmentation)[i][0],
                     sizeof(uint32_t)*(*mSegmentation)[i].size());

      //output.write((const char*)(mCoordinates->size()/3),sizeof(LocalIndexType));
      //output.write((const char*)&(*mCoordinates)[0], sizeof(FunctionType)*mCoordinates->size());
    }
  }
  else if ((mFlatSegmentation != NULL) && (mOffsets != NULL)) {

    if (mASCIIFlag) {
      for (i=0;i<=mSegCount;i++)
        output << (*mOffsets)[i] << std::endl;

      for (i=0;i<mSegCount;i++) {
        for (uint32_t k=(*mOffsets)[i];k<(*mOffsets)[i+1];k++)
          output << (*mFlatSegmentation)[k] << " ";
        output << std::endl;
      }
      //TODO add ASCII output for mSteepest
    }
    else {
      output.write((const char*)&(*mOffsets)[0],sizeof(uint32_t)*(mSegCount+1));
      output.write((const char*)&(*mFlatSegmentation)[0],sizeof(uint32_t)*mFlatSegmentation->size());
      #ifdef ENABLE_STEEPEST
      output.write((const char*)&(*mSteepest)[0],sizeof(uint32_t)*mSteepest->size());
      #endif
    }
  }

  output.write((const char*)&(mIndexMap)[0],sizeof(uint32_t)*mIndexMap.size());

  return 1;
}

}
