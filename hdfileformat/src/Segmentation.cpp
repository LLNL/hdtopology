/*
 * Segmentation.cpp
 *
 *  Created on: Feb 13, 2015
 *      Author: bremer5
 */

#include "Segmentation.h"
#include <string.h> //for memcpy

namespace HDFileFormat {

Segmentation::Segmentation() : mSegCount(0)
{
}

int Segmentation::initialize(SegmentationHandle& handle)
{
  std::vector<uint32_t> index_map;
#ifdef ENABLE_STEEPEST
  handle.readSegmentation(mOffsets,mSegmentation,index_map, mSteepest);
#else
  handle.readSegmentation(mOffsets,mSegmentation,index_map);
#endif
  mSegCount = mOffsets.size()-1;

  if (index_map.size() > 0)
    setIndexMap(&index_map);

  return 1;
}

int Segmentation::setSegmentation(const std::vector<std::vector<uint32_t> >* seg)
{
  std::vector<std::vector<uint32_t> >::const_iterator it;
  uint32_t count = 0;
  uint32_t i = 1;

  mOffsets.resize(seg->size()+1);

  mOffsets[0] = 0;
  for (it=seg->begin();it!=seg->end();it++) {
    count += it->size();
    mOffsets[i++] = count;
  }

  mSegCount = seg->size();

  mSegmentation.resize(count);
  for (i=0;i<mOffsets.size()-1;i++)
    memcpy(&mSegmentation[mOffsets[i]],&(*seg)[i][0],(*seg)[i].size()*sizeof(uint32_t));

  return 1;
}

#ifdef ENABLE_STEEPEST
int Segmentation::setSteepest(const std::vector<uint32_t> *steepest)
{
  mSteepest.resize(steepest->size());
  std::copy(&(*steepest)[0], &(*steepest)[0]+steepest->size(), mSteepest.begin());

  return 1;
}
#endif

int Segmentation::setIndexMap(const std::vector<uint32_t>* index)
{
  std::vector<uint32_t>::const_iterator it;
  uint32_t count = 0;

  mIndexMap.clear();
  for (it=index->begin();it!=index->end();it++) {
    //fprintf(stderr,"Segmentation::setIndexMap adding %d\n",*it);
    mIndexMap[*it] = count++;
  }

  return 1;
}


Segment Segmentation::elementSegmentation(uint32_t global_id) const
{
  Segment segment;

  //debug
  std::vector<uint32_t> keys;
  for(IndexMap::const_iterator it = mIndexMap.begin(); it != mIndexMap.end(); ++it)
  {
    keys.push_back(it->first);
    //cout << it->first << "\n";
  }

  uint32_t l = local(global_id);

  segment.samples = &mSegmentation[mOffsets[l]];
  segment.size = (int32_t)mOffsets[l+1] - (int32_t)mOffsets[l];

  return segment;
}

SegmentationHandle Segmentation::makeHandle()
{
  SegmentationHandle handle;

  handle.setOffsets(&mOffsets);
  handle.setSegmentation(&mSegmentation);

  if (!mIndexMap.empty()) {

    std::vector<uint32_t> index_map(mSegCount);
    IndexMap::const_iterator it;
    for (it=mIndexMap.begin();it!=mIndexMap.end();it++) {
      //fprintf(stderr,"Trying to set %d  %d\n",it->second,index_map.size());
      index_map[it->second] = it->first;
    }

    handle.setIndexMap(index_map);
  }

  return handle;
}

uint32_t Segmentation::local(uint32_t id) const
{
  if (mIndexMap.size() > 0) {
    IndexMap::const_iterator it;
    it = mIndexMap.find(id);

    hderror(it==mIndexMap.end(),"Have an index map but could not find global index %d",id);

    return it->second;
  }
  else
    return id;

}

}
