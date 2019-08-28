/*
 * DataCollectionHandle.h
 *
 *  Created on: Aug 20, 2014
 *      Author: bremer5
 */

#ifndef DATACOLLECTIONHANDLE_H
#define DATACOLLECTIONHANDLE_H

#include <string>
#include <cassert>
#include "FileHandle.h"
#include "DatasetHandle.h"

namespace HDFileFormat {

class DataCollectionHandle : public FileHandle
{
public:

  //! Current major version number
  static const uint16_t sMajorVersion;

  //! Current minor version number
  static const uint16_t sMinorVersion;

  //! ASCII floating point precision
  static const uint8_t sPrecision;

  //! The default name of the collection
  static const std::string sDefaultName;

  //! Default constructor
  DataCollectionHandle();

  //! Constructor linked to a file
  explicit DataCollectionHandle(const std::string& filename);

  //! Constructor linked to a file
  explicit DataCollectionHandle(const char* filename);

  //! Copy constructor
  DataCollectionHandle(const DataCollectionHandle& handle);

  //! Destructor
  ~DataCollectionHandle();

  //! Assignment operator
  DataCollectionHandle& operator=(const DataCollectionHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new DataCollectionHandle(*this);}

  //! Return the name of this collection
  std::string collection() const {return mCollection;}

  //! Return the number of datasets in this collection
  uint32_t datasetCount() const {return uint32_t(mDatasets.size());}

  //! Return the i'th dataset
  DatasetHandle& dataset(int i)
  {
    assert(i<int(mDatasets.size()) );
    return mDatasets[i];
  }

  //! Return the i'th dataset
  const DatasetHandle& dataset(int i) const
  {assert(i<int(mDatasets.size())); return mDatasets[i];}

  //! Connect a handle to the given filename
  int attach(const std::string& filename) {return attach(filename.c_str());}

  //! Connect a handle to the given filename
  int attach(const char* filename);

  //! Write the family to the given filename
  void write(const char* filename = NULL);

  //! Append the given handle (and its data) to the file
  void append(DatasetHandle& handle) {add(handle);appendData(mDatasets.back());}

  //! Add the given handle to the internal data structure but don't write
  virtual FileHandle& add(const FileHandle& handle);

  //! Update the XML-Footer
  void updateMetaData(const char* filename=NULL);

protected:

  //! The name of this collection
  std::string mCollection;

  //! The major version number
  uint16_t mMajor;

  //! The minor version number
  uint16_t mMinor;

  //! The collection of datasets or rather their handles
  std::vector<DatasetHandle> mDatasets;

  //! Reset all values to their default uninitialized values
  void clear();

  //! Append the data of the given handle to the file and re-write the footer
  void appendData(FileHandle& handle);
  //virtual void appendData(DatasetHandle& handle);

  //! Parse the xml tree
  virtual int parseXML(const XMLNode& node);

  //! Parse the local information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Attach the xml tree
  virtual int attachXML(XMLNode& node) const;

  //! Attach the local variables to the given node
  virtual int attachXMLInternal(XMLNode& node) const;

  //! Write the XML-footer to the given stream
  int attachXMLFooter(std::ofstream& output);
};

}


#endif /* DATACOLLECTIONHANDLE_H_ */
