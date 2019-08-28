/*
 * HierarchicalSegmentation.cpp
 *
 *  Created on: Feb 13, 2015
 *      Author: bremer5
 */

#include "HierarchicalSegmentation.h"

namespace HDFileFormat {

HierarchicalSegmentation::HierarchicalSegmentation()
{
}

int HierarchicalSegmentation::initialize(SegmentationHandle& handle)
{
  Segmentation::initialize(handle);

  DataBlockHandle* hierarchy = handle.getChildByType<DataBlockHandle>("Hierarchy");

  hderror(hierarchy==NULL,"No hierarchy found");

  mHierarchy.resize(hierarchy->sampleCount());

  hierarchy->readData(&mHierarchy[0]);


  return 1;
}

SegmentationHandle HierarchicalSegmentation::makeHandle()
{
  SegmentationHandle handle;

  handle = Segmentation::makeHandle();

  DataBlockHandle block;

  block.idString("Hierarchy");
  block.setData<MergeInfo>(&mHierarchy[0],mHierarchy.size());

  handle.add(block);
  fprintf(stderr,"Added Hierarchy %d\n",handle.childCount());

  return handle;
}

void HierarchicalSegmentation::addCancellation(uint32_t global_child, uint32_t global_parent, float parameter)
{
  uint32_t local_child;
  uint32_t local_parent;

  IndexMap::const_iterator it;

  //fprintf(stderr,"addCancellation %d %d %f\n",global_child,global_parent,parameter);
  it = mIndexMap.find(global_child);
  if (it == mIndexMap.end()) {
    hdwarning("Could not find child index %d in index map.",global_child);
    return;
  }
  local_child = it->second;

  it = mIndexMap.find(global_parent);
  if (it == mIndexMap.end()) {
    hdwarning("Could not find parent index %d in index map.",global_child);
    return;
  }
  local_parent = it->second;

  if (mHierarchy.size() <= local_child)
    mHierarchy.resize(local_child+1);

  mHierarchy[local_child].id = global_child;
  mHierarchy[local_child].parent = local_parent;
  mHierarchy[local_child].parameter = parameter;

}


int HierarchicalSegmentation::segmentation(Segmentation& segmentation, float parameter)
{
  std::vector<std::vector<uint32_t> > tmp;
  IndexMap index_map;
  IndexMap::iterator mIt;
  uint32_t rep; // Index of the representative
  uint32_t s; // Local index of the segment
  Segment seg;


  for (uint32_t i=0;i<mSegCount;i++) {
    seg = elementSegmentation(mHierarchy[i].id);

    // Get the current representative of this extremum
    rep = representative(i,parameter);

    // Check whether we have seen this representative before
    mIt = index_map.find(mHierarchy[rep].id);

    if (mIt == index_map.end()) { // If not
      index_map[mHierarchy[rep].id] = tmp.size(); // Create a new segement
      tmp.push_back(std::vector<uint32_t>());
      s = tmp.size()-1;
    }
    else {
      s  = mIt->second;
    }

    tmp[s].insert(tmp[s].end(),seg.samples,seg.samples+seg.size);
  }

  segmentation.setSegmentation(&tmp);
  segmentation.setIndexMap(index_map);

  return 1;
}

uint32_t HierarchicalSegmentation::representative(uint32_t index, float parameter) const
{
  while ((mHierarchy[index].parent != index) && (mHierarchy[index].parameter <= parameter))
    index = mHierarchy[index].parent;

  return index;
}



std::istream& operator>>(std::istream &input, HierarchicalSegmentation::MergeInfo &info)
{
  input >> info.id;
  input >> info.parent;
  input >> info.parameter;

  return input;
}

std::ostream& operator<<(std::ostream &output, HierarchicalSegmentation::MergeInfo &info)
{
  output << info.id << " " << info.parent << " " << info.parameter;

  return output;
}


}


