/*
 * DataHandle.h
 *
 *  Created on: Aug 19, 2014
 *      Author: bremer5
 */

#ifndef DATASETHANDLE_H
#define DATASETHANDLE_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>


#include "FileHandle.h"
#include "DataPointsHandle.h"
#include "FunctionHandle.h"

namespace HDFileFormat {

class DatasetHandle : public FileHandle
{
public:

  friend class DataCollectionHandle;

  //! The default name of the data set
  static const std::string sDefaultDatasetName;

  //! Constructor
  explicit DatasetHandle(HandleType t=H_DATASET);

  //! Constructor
  DatasetHandle(const char *filename, HandleType t=H_DATASET);

  //! Copy constructor
  DatasetHandle(const DatasetHandle& handle);

  //! Destructor
  virtual ~DatasetHandle();

  //! Assignment operator
  DatasetHandle& operator=(const DatasetHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new DatasetHandle(*this);}

  //! Add a child but do a type-check
  virtual FileHandle& add(const FileHandle& handle);
  
  //! Access to data points in the dataset, assuming the index 0 is the dataPointsHandle
  HDFileFormat::DataPointsHandle& dataPoints(){return *static_cast<DataPointsHandle*>(this->mChildren[0]);}

  //! Get a list of children by type
  std::vector<HDFileFormat::FunctionHandle> getFunctions() {std::vector<HDFileFormat::FunctionHandle> children;getAllChildrenByType<FunctionHandle>(children);return children;}

  //! Get a list of children by type
  HDFileFormat::FunctionHandle& getFunction(uint32_t i) {return *this->getChildByType<FunctionHandle>(i);}

};


}



#endif /* DATAHANDLE_H_ */
