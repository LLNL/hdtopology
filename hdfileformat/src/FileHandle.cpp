#include <errno.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cstdio>

#include "FileHandle.h"
#include "DataCollectionHandle.h"
//#include "GroupHandle.h"
#include "ClusterHandle.h"
#include "DatasetHandle.h"
#include "DataPointsHandle.h"
#include "DataPointsMetaInfoHandle.h"
#include "EmbeddingHandle.h"
#include "SubspaceHandle.h"
#include "HierarchyHandle.h"
#include "BasisHandle.h"
#include "GraphHandle.h"
#include "SegmentationHandle.h"
#include "FunctionHandle.h"
#include "HistogramHandle.h"
#include "DistributionHandle.h"

// #include "VolumeSegmentHandle.h"

namespace HDFileFormat {

const char* gHandleNames[HANDLE_COUNT] = {
  "Collection",
  "Dataset",

  "DataBlock",
  "DataPoints",
  "Cluster",
  "Embedding",
  "Graph",
  "Segmentation",
  "Function",
  "Hierarchy",
  "Subspace",
  "Basis",

  "VolumeSegment",
  "DataPointsMetaInfo",
  "Histogram",
  "Distribution",

  "Undefined"
};

const std::string FileHandle::sEmptyString = std::string("");

const char FileHandle::sStringSeperator = ',';

const char* FileHandle::typeName(HandleType t)
{
  hderror(t==H_UNDEFINED,"Handle type not set cannot determine name.");

  return gHandleNames[t];
}


FileHandle* FileHandle::constructHandle(const char* name, const std::string& filename)
{
  return constructHandle(name,filename.c_str());
}


FileHandle* FileHandle::constructHandle(const char* name,const char* filename)
{

  if (strcmp(name,gHandleNames[H_COLLECTION]) == 0)
    return new DataCollectionHandle(filename);
  //else if (strcmp(name,gHandleNames[H_GROUP]) == 0)
  //  return new GroupHandle(filename);
  else if (strcmp(name,gHandleNames[H_DATASET]) == 0)
    return new DatasetHandle(filename);

  else if (strcmp(name,gHandleNames[H_DATABLOCK]) == 0)
    return new DataBlockHandle(filename);
  else if (strcmp(name,gHandleNames[H_DATAPOINTS]) == 0)
    return new DataPointsHandle(filename);
  else if (strcmp(name,gHandleNames[H_CLUSTER]) == 0)
    return new ClusterHandle(filename);
  else if (strcmp(name,gHandleNames[H_EMBEDDING]) == 0)
    return new EmbeddingHandle(filename);
  else if (strcmp(name,gHandleNames[H_GRAPH]) == 0)
    return new GraphHandle(filename);
  else if (strcmp(name,gHandleNames[H_SEGMENTATION]) == 0)
    return new SegmentationHandle(filename);
  else if (strcmp(name,gHandleNames[H_FUNCTION]) == 0)
    return new FunctionHandle(filename);
  else if (strcmp(name,gHandleNames[H_HIERARCHY]) == 0)
    return new HierarchyHandle(filename);
  else if (strcmp(name,gHandleNames[H_SUBSPACE]) == 0)
    return new SubspaceHandle(filename);
  else if (strcmp(name,gHandleNames[H_BASIS]) == 0)
    return new BasisHandle(filename);
  // else if(strcmp(name,gHandleNames[H_VOLSEGMENT]) == 0)
  //   return new VolumeSegmentHandle(filename);
  else if(strcmp(name,gHandleNames[H_DATAPOINTS_METAINFO]) == 0)
    return new DataPointsMetaInfoHandle(filename);
  else if(strcmp(name,gHandleNames[H_HISTOGRAM]) == 0)
    return new HistogramHandle(filename);
  else if(strcmp(name,gHandleNames[H_DISTRIBUTION]) == 0)
    return new DistributionHandle(filename);

    hderror(true,"Unkown handle name \"%s\".",name);

  return NULL;
}


FileHandle::FileHandle(HandleType t)
  :mType(t),
   mFileName(""),
   mOffset(-1),
   mTopHandle(NULL),
   mASCIIFlag(false)
{
}

FileHandle::FileHandle(const char* filename, HandleType t)
  :mType(t),
   mFileName(filename),
   mOffset(-1),
   mTopHandle(NULL),
   mASCIIFlag(false)
{
}

FileHandle::FileHandle(const FileHandle& handle)
  :mType(handle.mType),
   mFileName(handle.mFileName),
   mOffset(handle.mOffset),
   mTopHandle(handle.mTopHandle),
   mASCIIFlag(handle.mASCIIFlag),
   mID(handle.mID)
{
  mChildren.resize(handle.mChildren.size(),NULL);

  for (uint32_t i=0;i<mChildren.size();i++) {
    mChildren[i] = handle.mChildren[i]->clone();
  }
}

FileHandle::~FileHandle()
{
  for (uint32_t i=0;i<mChildren.size();i++)
  {
    if(mChildren[i])
      delete mChildren[i];
  }
}

FileHandle& FileHandle::operator=(const FileHandle& handle)
{
  //Error in release build (Shusen Liu)
  //printf("FileHandle: %s = %s\n", this->typeName(), handle.typeName());
  assert(mType == handle.mType); //for the debug case
  hderror(mType!=handle.mType,"Assignment between incompatible types.");

  mFileName = handle.mFileName;

  mOffset = handle.mOffset;

  mTopHandle = handle.mTopHandle;
  mASCIIFlag = handle.mASCIIFlag;
  mID = handle.mID;

  mChildren.resize(handle.mChildren.size(),NULL);

  for (uint32_t i=0;i<mChildren.size();i++) {
    mChildren[i] = handle.mChildren[i]->clone();
  }

  return *this;
}

void FileHandle::instantiate()
{
  for(size_t i=0; i<mChildren.size(); i++)
    mChildren[i]->instantiate();
}

const char* FileHandle::typeName() const
{
  hderror(mType==H_UNDEFINED,"Handle type not set cannot determine name.");

  return gHandleNames[mType];
}

FileHandle& FileHandle::add(const FileHandle& handle)
{
  FileHandle *handleClone = handle.clone();

  //change the top handle to current one
  handleClone->topHandle(this->topHandle());

  mChildren.push_back(handleClone);

  //fprintf(stderr,"FileHandle::add \"%s\"\n",handle.typeName());
  return *mChildren.back();
}


DataPointsHandle& FileHandle::getDataPoints(uint32_t i)
{
  return *this->getChildByType<DataPointsHandle>(i);
}

DataBlockHandle& FileHandle::getDataBlock(uint32_t i)
{
  return *this->getChildByType<DataBlockHandle>(i);
}

GraphHandle& FileHandle::getGraph(uint32_t i)
{
  return *this->getChildByType<GraphHandle>(i);
}

SubspaceHandle &FileHandle::getSubspace(uint32_t i)
{
  return *this->getChildByType<SubspaceHandle>(i);
}

BasisHandle &FileHandle::getBasis(uint32_t i)
{
  return *this->getChildByType<BasisHandle>(i);
}

ClusterHandle &FileHandle::getCluster(uint32_t i)
{
  return *this->getChildByType<ClusterHandle>(i);
}

ClusterHandle &FileHandle::getClusterByName(std::string clusterName)
{
  return *this->getChildByType<ClusterHandle>(clusterName);
}

void FileHandle::clear()
{
  mFileName = std::string("");

  mOffset = 0;
  mTopHandle = NULL;  //! Add the attribute to the node

}

void FileHandle::appendData(FileHandle& handle)
{
  hderror(mType!=H_COLLECTION,"Sorry, append can only be called from a DataCollectionHandle");
}

int FileHandle::writeData(std::ofstream &output, const std::string &filename)
{
  this->mFileName = filename;
  this->mOffset = static_cast<FileOffsetType>(output.tellp());

  //write handle's own data
  this->writeDataInternal(output, filename);

  //write handle's serialized data
  this->writeSerializeData(output);

  //write the data in the children handle
  for(size_t i=0; i<this->mChildren.size(); i++)
    mChildren[i]->writeData(output, filename);

  return 1;

}

int FileHandle::parseXML(const XMLNode &node)
{
  FileHandle* handle;

  this->parseXMLInternal(node);

  for (int32_t i=0;i<node.nChildNode();i++) {

    handle = this->constructHandle(node.getChildNode(i).getName(),mFileName);
    handle->topHandle(topHandle());

    handle->parseXML(node.getChildNode(i));
    mChildren.push_back(handle);
  }

  return 1;
}


int FileHandle::parseXMLInternal(const XMLNode& node)
{
  uint64_t tmp;

  getAttribute(node,"offset",tmp);
  getAttribute(node,"encoding",mASCIIFlag);
  getAttribute(node,"name",mID);

  mOffset = static_cast<std::streamoff>(tmp);

  return 1;
}

int FileHandle::attachXML(XMLNode &node) const
{
  //fprintf(stderr,"FileHandle::attachXML \"%s\"\n",this->typeName());
  //attach internal
  this->attachXMLInternal(node);

  //attach the tree
  XMLNode child;
  for (uint8_t i=0;i<mChildren.size();i++) {

    child = node.addChild(mChildren[i]->typeName());
    mChildren[i]->attachXML(child);
  }

  return 1;
}


int FileHandle::attachXMLInternal(XMLNode& node) const
{
  //fprintf(stderr,"FileHandle::attachXMLInternal \"%s\"\n",this->typeName());

  addAttribute(node,"offset",mOffset);
  addAttribute(node,"encoding",mASCIIFlag);
  addAttribute(node,"name",mID);

  return 1;
}

int FileHandle::rewind(std::ifstream& file) const
{
  if (!file.good()) {
    fprintf(stderr,"Cannot rewind NULL file pointer.");
    return 0;
  }

  file.seekg(mOffset,std::ios_base::beg);

  if (!file.good()) {
    fprintf(stderr,"Error rewinding the file.");
    return 0;
  }

  return 1;
}

int FileHandle::openOutputFile(const std::string& filename, std::ofstream& file,
                               bool binary) const
{
  return openOutputFile(filename.c_str(),file,binary);
}

int FileHandle::openOutputFile(const char* filename, std::ofstream& file,
                               bool binary) const
{
  if (strcmp(filename,"")==0) {
    fprintf(stderr,"Cannot open an empty file name \"\"");
    return 0;
  }

  if (binary)
  {
    file.open(filename,std::ios::out | std::ios::binary);
  }
  else
  {
    file.open(filename,std::ios::out);
  }

  if (file.fail()) {
    fprintf(stderr,"Could not open file \"%s\". Got errno %d = \"%s\".\n",filename,errno,strerror(errno));
    return 0;
  }

  return 1;
}

int FileHandle::openOutputFileToAppend(const std::string& filename, std::ofstream& file,
                               bool binary) const
{
  return openOutputFileToAppend(filename.c_str(),file,binary);
}

int FileHandle::openOutputFileToAppend(const char* filename, std::ofstream& file,
                               bool binary) const
{
  if (strcmp(filename,"")==0) {
    fprintf(stderr,"Cannot open an empty file name \"\"");
    return 0;
  }

  if (binary)
  {
    file.open(filename,std::ios::in | std::ios::out | std::ios::binary);
  }
  else
  {
    file.open(filename,std::ios::in | std::ios::out);
  }

  if (file.fail()) {
    fprintf(stderr,"Could not open file \"%s\". Got errno %d = \"%s\".\n",filename,errno,strerror(errno));
    return 0;
  }

  return 1;
}

int FileHandle::openInputFile(const std::string& filename, std::ifstream& file, bool binary) const
{
  return openInputFile(filename.c_str(),file,binary);
}

int FileHandle::openInputFile(const char* filename, std::ifstream& file, bool binary) const
{
  if (strcmp(filename,"") == 0) {
    fprintf(stderr,"Cannot open an empty file name \"\"");
    return 0;
  }

  if (binary)
    file.open(filename, std::ios::in | std::ios::binary);
  else
    file.open(filename, std::ios::in);


  //std::cout << "file.fail() = " << file.fail() << std::endl;

  // JCB: crashes when you try to print mode as a string
  //hderror(true,"Could not open file \"%s\". Got errno %d = \"%s\".\n",filename,errno,strerror(errno));
  if (!file.good()) {
    fprintf(stderr,"Could not open file \"%s\" . Got errno %d = \"%s\".\n",filename,errno,strerror(errno));
    return 0;
  }

  return 1;
}

std::vector<std::string> FileHandle::splitString(const std::string& input,const char sep)
{
  std::vector<std::string> tokens;
  std::istringstream str(input);
  std::string token;

  while(getline(str, token, sep))
  {
    tokens.push_back(token);
  }

  return tokens;
}

int FileHandle::getChildrenCountByType(const char *typeName)
{
  int childrenCount = 0;
  for (int32_t i=0;i<mChildren.size();i++)
    if(std::string(typeName) == mChildren[i]->typeName())
      childrenCount++;
  return childrenCount;
}

int FileHandle::getChildrenCountByType(HandleType type)
{
  int childrenCount = 0;
  for (int32_t i=0;i<mChildren.size();i++)
    if(type == mChildren[i]->type())
      childrenCount++;
  return childrenCount;
}

//! Template specialization for getting different type of first child
//! clone here is necessary, since you don't want to different object share the same pointer

//! DataBlockHandle
template<>
void FileHandle::getFirstChildByType(DataBlockHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DATABLOCK)
      handleOutput = *dynamic_cast<DataBlockHandle*>(mChildren[i]->clone());
}

//! DataPointsHandle
template<>
void FileHandle::getFirstChildByType(DataPointsHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DATAPOINTS)
      handleOutput = *dynamic_cast<DataPointsHandle*>(mChildren[i]->clone());
}


//! ClusterHandle
template<>
void FileHandle::getFirstChildByType(ClusterHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_CLUSTER)
      handleOutput = *dynamic_cast<ClusterHandle*>(mChildren[i]->clone());
}

//! Subspace
template<>
void FileHandle::getFirstChildByType(SubspaceHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_SUBSPACE)
      handleOutput = *dynamic_cast<SubspaceHandle*>(mChildren[i]->clone());
}

//! Basis
template<>
void FileHandle::getFirstChildByType(BasisHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_BASIS)
      handleOutput = *dynamic_cast<BasisHandle*>(mChildren[i]->clone());
}

//! HierarchyHandle
template<>
void FileHandle::getFirstChildByType(HierarchyHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_HIERARCHY)
      handleOutput = *dynamic_cast<HierarchyHandle*>(mChildren[i]->clone());
}

//! EmbeddingHandle
template<>
void FileHandle::getFirstChildByType(EmbeddingHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_EMBEDDING)
      handleOutput = *dynamic_cast<EmbeddingHandle*>(mChildren[i]->clone());
}

//! NeighborhoodGraphHandle
template<>
void FileHandle::getFirstChildByType(GraphHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_GRAPH)
      handleOutput = *dynamic_cast<GraphHandle*>(mChildren[i]->clone());
}

//! VolumeSegmentHandle
// template<>
// void FileHandle::getFirstChildByType(VolumeSegmentHandle &handleOutput)
// {
//   for(size_t i=0; i<mChildren.size(); i++)
//     if(mChildren[i]->type()==H_VOLSEGMENT)
//       handleOutput = *dynamic_cast<VolumeSegmentHandle*>(mChildren[i]->clone());
// }

//! DataPointsMetaInfoHandle
template<>
void FileHandle::getFirstChildByType(DataPointsMetaInfoHandle &handleOutput)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DATAPOINTS_METAINFO)
      handleOutput = *dynamic_cast<DataPointsMetaInfoHandle*>(mChildren[i]->clone());
}

///////////////////////////////////////////////////////////////

//! Template specialization for getting different type of all children
//! DataBlockHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<DataBlockHandle> &childrenVec)
{
  childrenVec.clear();

  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DATABLOCK)
      childrenVec.push_back( *dynamic_cast<DataBlockHandle*>(mChildren[i]->clone()));
}

//! DatasetHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<DatasetHandle> &childrenVec)
{
  childrenVec.clear();

  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DATASET)
      childrenVec.push_back( *dynamic_cast<DatasetHandle*>(mChildren[i]->clone()));
}

//! DataPointsHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<DataPointsHandle> &childrenVec)
{
  childrenVec.clear();

  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DATAPOINTS)
      childrenVec.push_back( *dynamic_cast<DataPointsHandle*>(mChildren[i]->clone()));
}

//! ClusterHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<ClusterHandle> &childrenVec)
{
  childrenVec.clear();

  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_CLUSTER)
      childrenVec.push_back( *dynamic_cast<ClusterHandle*>(mChildren[i]->clone()));
}


//! Subspace
template<>
void FileHandle::getAllChildrenByType(std::vector<SubspaceHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_SUBSPACE)
      childrenVec.push_back( *dynamic_cast<SubspaceHandle*>(mChildren[i]->clone()));
}

//! Basis
template<>
void FileHandle::getAllChildrenByType(std::vector<BasisHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_BASIS)
      childrenVec.push_back( *dynamic_cast<BasisHandle*>(mChildren[i]->clone()));
}

//! HierarchyHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<HierarchyHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_HIERARCHY)
      childrenVec.push_back( *dynamic_cast<HierarchyHandle*>(mChildren[i]->clone()));
}

//! EmbeddingHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<EmbeddingHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_EMBEDDING)
      childrenVec.push_back( *dynamic_cast<EmbeddingHandle*>(mChildren[i]->clone()));
}

//! NeighborhoodGraphHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<GraphHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_GRAPH)
      childrenVec.push_back( *dynamic_cast<GraphHandle*>(mChildren[i]->clone()));
}

//! DistributionHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<DistributionHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_DISTRIBUTION)
      childrenVec.push_back( *dynamic_cast<DistributionHandle*>(mChildren[i]->clone()));
}

//! HistogramHandle
template<>
void FileHandle::getAllChildrenByType(std::vector<HistogramHandle> &childrenVec)
{
  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type()==H_HISTOGRAM)
      childrenVec.push_back( *dynamic_cast<HistogramHandle*>(mChildren[i]->clone()));
}
//! VolumeSegmentHandle
// template<>
// void FileHandle::getAllChildrenByType(std::vector<VolumeSegmentHandle> &childrenVec)
// {
//   for(size_t i=0; i<mChildren.size(); i++)
//     if(mChildren[i]->type()==H_VOLSEGMENT)
//       childrenVec.push_back( *dynamic_cast<VolumeSegmentHandle*>(mChildren[i]->clone()));
// }




} //namespace
