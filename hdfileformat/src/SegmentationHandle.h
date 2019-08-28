/*
 * SegmentationHandle.h
 *
 *  Created on: Feb 15, 2012
 *      Author: bremer5
 */

#ifndef SEGMENTATIONHANDLE_H_
#define SEGMENTATIONHANDLE_H_

#include "FileHandle.h"
#include "DataBlockHandle.h"

namespace HDFileFormat{


//! A class that encapsulates the handle to a segmentation
class SegmentationHandle : public FileHandle
{
public:

  //! Default constructor
  SegmentationHandle();

  //! Constructor
  SegmentationHandle(const char* filename);

  //! Copy constructor
  SegmentationHandle(const SegmentationHandle& handle);

  //! Destructor
  virtual ~SegmentationHandle();

  //! Assignment operator
  SegmentationHandle& operator=(const SegmentationHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new SegmentationHandle(*this);}

  //! Return the type of segmentation we could create
  std::string segmentationType() const;

  /***********************************************************************
   ************************  Access to members ***************************
   ***********************************************************************/

  //! Return the number of segments
  uint32_t segCount() const {return mSegCount;}

  //! Set the segmentation data as set of sets
  void setSegmentation(std::vector<std::vector<uint32_t> >* segmentation);

  //! Set the offsets
  void setOffsets(std::vector<uint32_t>* offsets);

  //! Set the segmentation as flat array
  void setSegmentation(std::vector<uint32_t>* segmentation);

  //! Set the steepest
  void setSteepest(std::vector<uint32_t>* steepest);

  //! Set the index map (not that the index map is owned by us)
  void setIndexMap(std::vector<uint32_t>& index_map);

  /*******************************************************************************************
   ****************************  Interface to internal handles *******************************
   ******************************************************************************************/

  //! Add the given handle to the internal data structure but don't write
  virtual FileHandle& add(const FileHandle& handle);

  /*******************************************************************************************
   ****************************  Interface to read the data    *******************************
   ******************************************************************************************/
#ifdef ENABLE_STEEPEST
  //! Read the raw segmentation data
  int readSegmentation(std::vector<uint32_t>& offsets, std::vector<uint32_t>& segmentation,
                       std::vector<uint32_t>& index_map, std::vector<uint32_t> &steepest);
#else
  //! Read the raw segmentation data
  int readSegmentation(std::vector<uint32_t>& offsets, std::vector<uint32_t>& segmentation,
                       std::vector<uint32_t>& index_map);
#endif
protected:

   //! The number of segments should be identical to mSegmentation->size()
  uint32_t mSegCount;

  //! The number of total elements of the segmentation
  uint32_t mSegSize;

  //! FLag to indicate whether this segmetation contains an index map
  bool mHasIndexMap;

  //! The segmentation information as sets of sets
  std::vector<std::vector<uint32_t> >* mSegmentation;

  //! The segmentation information as flat array
  std::vector<uint32_t>* mFlatSegmentation;

  //! steepest storing the ridge line information
  std::vector<uint32_t>* mSteepest;

  //! The array of offsets corresponding to the flat array
  std::vector<uint32_t>* mOffsets;

  //! The index map for this segmentation
  std::vector<uint32_t> mIndexMap;

  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

  //! Write the local data
  virtual int writeDataInternal(std::ofstream& output, const std::string& filename);
};



}

#endif /* SEGMENTATIONHANDLE_H_ */
