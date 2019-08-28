/*
 * HierarchicalSegmentation.h
 *
 *  Created on: Feb 12, 2015
 *      Author: bremer5
 */

#ifndef HIERARCHICALSEGEMNTATION_H
#define HIERARCHICALSEGEMNTATION_H

#include "Segmentation.h"

namespace HDFileFormat {


class HierarchicalSegmentation : public Segmentation
{
public:

  //! Constructor
  HierarchicalSegmentation();

  //! Destructor
  virtual ~HierarchicalSegmentation() {}

  //! Initialize the segmentation from a segmentation handle
  int initialize(SegmentationHandle& handle);

  //! Construct a segmentation handle
  virtual SegmentationHandle makeHandle();

  //! Output a segmentation for a given parameter
  int segmentation(Segmentation& seg, float parameter);

  //! Add cancellation that merges the child with its parent at the given parameter
  void addCancellation(uint32_t global_child, uint32_t global_parent, float parameter);

  //! Set the order of cancellations
  void setOrder(const std::vector<uint32_t>* order) {mOrder = *order;}

//protected:

  //! The information for merged segments
  class MergeInfo {
  public:

    uint32_t id; //! Global id of this node
    uint32_t parent; //! Local index of the parent
    float parameter; //! Parameter threshold

    friend std::istream& operator>>(std::istream &input, MergeInfo &info);
    friend std::ostream& operator<<(std::ostream &output, MergeInfo &info);

  };
  //need to be a friend of the outter class as well
  friend std::istream& operator>>(std::istream &input, MergeInfo &info);
  friend std::ostream& operator<<(std::ostream &output, MergeInfo &info);

  //! A list of merge infos for all segments
  std::vector<MergeInfo> mHierarchy;

  //! The sorted list of segment id's smallest thresholds first
  std::vector<uint32_t> mOrder;

  //! Return the representative of a given local index
  uint32_t representative(uint32_t index, float parameter) const;
};

#ifndef SWIG
std::istream& operator>>(std::istream &input, HierarchicalSegmentation::MergeInfo &info);
std::ostream& operator<<(std::ostream &output, HierarchicalSegmentation::MergeInfo &info);
#endif
}


#endif


