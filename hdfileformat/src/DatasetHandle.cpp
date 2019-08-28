/*
 * DataHandle.cpp
 *
 *  Created on: Aug 19, 2014
 *      Author: bremer5
 */

#include "DatasetHandle.h"

namespace HDFileFormat {

const std::string DatasetHandle::sDefaultDatasetName = "HDData";


DatasetHandle::DatasetHandle(HandleType t)
  :FileHandle(t)
{
  mID = sDefaultDatasetName;
}

DatasetHandle::DatasetHandle(const char *filename, HandleType t)
  :FileHandle(filename,t)
{
  mID = sDefaultDatasetName;
}

DatasetHandle::DatasetHandle(const DatasetHandle& handle)
  :FileHandle(handle)
{

}

DatasetHandle::~DatasetHandle()
{

}

DatasetHandle& DatasetHandle::operator=(const DatasetHandle& handle)
{
  FileHandle::operator=(handle);

  return *this;
}

FileHandle& DatasetHandle::add(const FileHandle& handle)
{
  switch (handle.type()) {
    case H_DATAPOINTS:
      return FileHandle::add(handle);
      break;
    case H_CLUSTER:
      return FileHandle::add(handle);
      break;
    case H_EMBEDDING:
      return FileHandle::add(handle);
      break;
    case H_GRAPH:
      return FileHandle::add(handle);
      break;
    case H_FUNCTION:
      return FileHandle::add(handle);
      break;
    case H_VOLSEGMENT:
      return FileHandle::add(handle);
      break;
    case H_COLLECTION:
    case H_BASIS:
    case H_SUBSPACE:
    case H_DATABLOCK: //allow datablock
      return FileHandle::add(handle);
      break;
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside datasets.",handle.typeName());
      break;
  }
  //should never reach here
  return *this;
}


}

