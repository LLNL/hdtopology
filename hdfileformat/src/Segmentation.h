/*
 * Segmentation.h
 *
 *  Created on: Feb 12, 2015
 *      Author: bremer5
 */
#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <vector>
#include <map>

//#include "Definitions.h"
#include "SegmentationHandle.h"

namespace HDFileFormat {

//! A segment encapsulates a pointer to a list of indices and the
//! number of indices present
struct Segment {
  uint32_t size;
  const uint32_t* samples;

  Segment() : size(0),samples(NULL) {}
};


//! Baseclass to hold a segmentation
class Segmentation
{
public:

  typedef std::map<uint32_t,uint32_t> IndexMap;

  //! Default construction
  Segmentation();

  //! Destructor
  virtual ~Segmentation() {}

  //! Initialize the segmentation from a segmentation handle
  int initialize(SegmentationHandle& handle);

  //! Set the segmentation using a set of sets
  int setSegmentation(const std::vector<std::vector<uint32_t> >* seg);

#ifdef ENABLE_STEEPEST
  //! Set the Steepest using a set of sets
  int setSteepest(const std::vector<uint32_t> *steepest);
#endif

  //! Set the index map
  int setIndexMap(const std::vector<uint32_t>* index);

#ifndef SWIG
  //! Set the index map
  int setIndexMap(const std::map<uint32_t,uint32_t>& index_map) {mIndexMap = index_map;return 1;}
#endif

  //! Return a reference to all points belonging to a segment
  Segment elementSegmentation(uint32_t seg_id) const;

  //! Construct a segmentation handle
  virtual SegmentationHandle makeHandle();

  //! Return the seg count
  uint32_t segCount() const {return mSegCount;}

  //! Return a pointer to the internal offset array
  const std::vector<uint32_t>* offsets() const {return &mOffsets;}

  //! Return a pointer to the internal segmentation array
  const std::vector<uint32_t>* segmentation() const {return &mSegmentation;}

//protected:
public:

  //! The number of segments
  uint32_t mSegCount;

  //! The segmentation as a list of point indices per label
  std::vector<uint32_t> mSegmentation;

  //! steepest
  std::vector<uint32_t> mSteepest;

  //! The list of offsets for labels
  std::vector<uint32_t> mOffsets;

  //! The set of valid representatives and their respective indices in the list of segmentations
  IndexMap mIndexMap;

  //! Convinience function to map the global to local index space
  uint32_t local(uint32_t id) const;
};

}

#endif
