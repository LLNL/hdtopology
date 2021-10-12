#include <map>
#include <cmath>
#include <limits>
#include <queue>
#include <algorithm>
#include "ExtremumGraph.h"
#include "EdgeIterator.h"
#include "NeighborhoodIterator.h"
#include <iostream>
#include <algorithm>

#include <DataPointsHandle.h>
#include <DistributionHandle.h>
#include <HistogramHandle.h>

#include <cereal/archives/binary.hpp>
//#include <cereal/archives/xml.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <membuf.h>

bool ExtremumGraphExt::cmp::operator()(uint32_t i, uint32_t j) const
{
  if (ascending) {
    if ((data->f(i) < data->f(j))
        || ((data->f(i) == data->f(j)) && (i < j)))
      return true;
  }
  else {
    if ((data->f(i) > data->f(j))
        || ((data->f(i) == data->f(j)) && (i > j)))
      return true;
  }

  return false;
}

bool ExtremumGraphExt::vertexCmp::operator()(uint32_t i, uint32_t j) const
{
  if (ascending) {
    if ((function[i] > function[j])
        || ((function[i] == function[j]) && (i > j)))
      return true;
  }
  else {
    if ((function[i] < function[j])
        || ((function[i] == function[j]) && (i < j)))
      return true;
  }

  return false;
}

bool ExtremumGraphExt::vertexCmp::operator()(uint32_t i, float f) const
{
  if (ascending)
    return (function[i] > f);
  else
    return (function[i] < f);
}

bool ExtremumGraphExt::vertexCmp::operator()(float f, uint32_t j) const
{
  if (ascending)
    return (f > function[j]);
  else
    return (f < function[j]);
}

ExtremumGraphExt::ExtremumGraphExt() : mRange(2)
{
  mAscending = true;
}
void ExtremumGraphExt::initialize(const HDData* data, const Flags* flags, const Neighborhood* edges,
                                  bool ascending, uint32_t max_segments,
                                  const ComputeMode mode, uint32_t cube_dim,
                                  uint32_t resolution, int32_t target_attr,
                                  std::vector<HistogramType> histogramTypes)
{
  //fprintf(stderr,"Number of edges %d\n",edges->size());

  if(edges){
    NeighborhoodIterator it(*edges);
    initialize(data, flags, it, ascending, max_segments, mode, cube_dim, resolution, target_attr, histogramTypes);
  }
  else{
    if(data)
      computeHistograms(data, cube_dim, resolution, histogramTypes, target_attr);
  }
}

void ExtremumGraphExt::initialize(const HDData* data,
                                  const Flags* flags,
                                  EdgeIterator& edgeIterator,
                                  bool ascending,
                                  uint32_t max_segments,
                                  const ComputeMode mode, uint32_t cube_dim,
                                  uint32_t resolution, int32_t target_attr,
                                  std::vector<HistogramType> histogramTypes)
{
  fprintf(stderr, "---- ExtremumGraphExt::initialize ---- %p %d\n",data,data->size());
  mAscending = ascending;
  fprintf(stderr, "  size: %u attr: %u dim: %u func: %u \n", data->size(), data->attr(), data->dim(), data->func());
  mFunction.resize(data->size());

  mRange[0] = 10e34;
  mRange[1] = -10e34;

  for (uint32_t i=0;i<mFunction.size();i++) {
    // We want to ignore all vertices that have been flagged as invalid
    if(flags){
        if ((*flags)[i]) {
          mFunction[i] = data->f(i);

          mRange[0] = std::min(mRange[0],mFunction[i]);
          mRange[1] = std::max(mRange[1],mFunction[i]);
        }
    }else{
        mFunction[i] = data->f(i);
        mRange[0] = std::min(mRange[0],mFunction[i]);
        mRange[1] = std::max(mRange[1],mFunction[i]);
    }
  }

  fprintf(stderr, "function min: %f function max: %f\n", mRange[0], mRange[1]);

  mExtrema.clear();
  mSaddles.clear();
  mSegments.clear();

  computeSegmentation(data, flags, edgeIterator, mode);
  computeHierarchy();

  sort(data);

  fprintf(stderr, "Before Simplify: mExtrema size: %ld, mSaddle size: %ld \n", mExtrema.size(), mSaddles.size());

  if (max_segments < mExtrema.size())
    simplify(max_segments);

  //! This would be temporary
  mPointLocations.clear();
  storeLocations(data);

  fprintf(stderr, "After simplification max_segment: %d, mExtrema size: %ld, mSaddle size: %ld \n", max_segments, mExtrema.size(), mSaddles.size());

  if ((mode == SEGMENTATION) || (mode == COMBINED)) {
    computeSegments(data);
  }


  if (mode==HISTOGRAM)
    mSegments.clear();

  if ((mode == HISTOGRAM) || (mode == COMBINED))
    computeHistograms(data, cube_dim, resolution, histogramTypes, target_attr);

  fprintf(stderr, "finish compute histogram \n");
}

uint32_t ExtremumGraphExt::countForPersistence(float persistence)
{
  unsigned int count=0;

  persistence *= (mRange[1] - mRange[0]);

  while ((count < mExtrema.size()) && (mExtrema[count].persistence > persistence))
    count++;

  return count;
}

std::vector<std::vector<uint32_t> > ExtremumGraphExt::segmentation(uint32_t count)
{
  // ! Segmentation method for Histogram

  if (mSegments.size()==0){
    // fprintf(stderr, "mSegments size is zero\n");
    return segmentHist(count);
  }

  uint32_t r;
  uint32_t pre_size;

  if (count == mSegCount){
    return mSegmentation;
  }

  mSegCount = std::min(count,(uint32_t)mExtrema.size());

  vertexCmp greater(mFunction,mAscending);


  mSegmentation.clear();

  // Here we check if mSegments is empty
  // Insert the top ten segments with highest p into mSegmentation

  mSegmentation.insert(mSegmentation.end(),mSegments.begin(),mSegments.begin()+mSegCount);

  {
    for (uint32_t i=mSegCount;i<mExtrema.size();i++) {
    r = rep(i,mSegCount);
    pre_size = mSegmentation[r].size();

    mSegmentation[r].insert(mSegmentation[r].end(),mSegments[i].begin(),mSegments[i].end());

    std::inplace_merge(mSegmentation[r].begin(),
                       mSegmentation[r].begin()+pre_size,
                       mSegmentation[r].end(),
                       greater);
  }

  // We need to guard against vertices with the same function value being mis-ordered
  // since the order of extrema is by persistence which is not guaranteed to preserve
  // order by function value (its a difference not a raw value)
    for (uint32_t i=0;i<mSegCount;i++) {
      if (mSegmentation[i][0] != mExtrema[i].id) {
      std::vector<uint32_t>::iterator it = std::find(mSegmentation[i].begin(),mSegmentation[i].end(),mExtrema[i].id);
      std::swap(mSegmentation[i][0],*it);
      }
    }
  }
  return mSegmentation;

}

std::vector<uint32_t> ExtremumGraphExt::segment(uint32_t ext, uint32_t count, float threshold)
{
  int32_t v = activeExtremum(ext,count);

  if (v < 0)
    return std::vector<uint32_t>();

  // Make sure we compute the segmentation for the right count
  segmentation(count);

  // Return histogram for histogram only mode
  if(mSegments.size()==0)
    return histogram(ext, count, threshold);

  vertexCmp greater(mFunction,mAscending);

  std::vector<uint32_t>::iterator last;
  last = std::upper_bound(mSegmentation[v].begin(),mSegmentation[v].end(),threshold,greater);

  std::vector<uint32_t> partial(mSegmentation[v].begin(),last);

  return partial;
}

// Will rewrite this
uint32_t ExtremumGraphExt::segmentSize(uint32_t ext, uint32_t count, float threshold)
{
  int32_t v = activeExtremum(ext,count);
  // fprintf(stderr, "activeExtremum: %d\n", v);

  if (v < 0)
    return 0;

  segmentation(count);

  //! segment size for histogram only mode
  if(mSegments.size()==0){
    // fprintf(stderr, "return histogram size\n");
    return histogramSize(ext, count, threshold);
  }

  vertexCmp greater(mFunction,mAscending);

  std::vector<uint32_t> seg = mSegmentation[v];
  std::vector<uint32_t>::iterator last;

  last = std::upper_bound(seg.begin(),seg.end(),threshold,greater);

  return (last - seg.begin());
 }


uint32_t ExtremumGraphExt::coreSize(uint32_t ext, uint32_t count)
{
  int32_t saddle;

  saddle = highestSaddleForExtremum(ext,count);
  if (saddle < 0)
    return 0;

  return segmentSize(ext,count,mFunction[saddle]);
}


std::vector<uint32_t> ExtremumGraphExt::coreSegment(uint32_t ext, uint32_t count)
{
  int32_t saddle;

  saddle = highestSaddleForExtremum(ext,count);

  return segment(ext,count,mFunction[saddle]);
}

int32_t ExtremumGraphExt::highestSaddleForExtremum(uint32_t ext, uint32_t count)
{
  int32_t v = activeExtremum(ext,count);

  if (v < 0)
    return -1;

  uint32_t left,right;
  vertexCmp greater(mFunction,mAscending);

  // Look at all saddles that could potentially exist
  std::vector<Saddle>::iterator sIt = mSaddles.begin();
  std::vector<Saddle>::iterator saddle = mSaddles.end();
  while ((sIt != mSaddles.end()) && (sIt->persistence >= mExtrema[count].persistence)) {

    // We only care about cancellation saddles
    if (!sIt->cancellation) {
      sIt++;
      continue;
    }

    // First find the active extremum on either side
    left = rep(sIt->neighbors[0],count);
    right = rep(sIt->neighbors[1],count);

    // IF this is not a strangulation and the saddle touches the extremum we are looking for
    if ((left != right) && ((left == v) || (right == v))) {

      if (saddle == mSaddles.end())
        saddle = sIt;
      else if (greater(sIt->id,saddle->id))
        saddle = sIt;
    }

    sIt++;
  }

  return saddle->id;
}



std::vector<float> ExtremumGraphExt::persistences() const
{
  std::vector<float> p(mExtrema.size());

  //don't need to divide the first itemize, since it will just be a very large number
  p[0] = mExtrema[0].persistence;

  for (uint32_t i=1;i<mExtrema.size();i++) {
    p[i] = mExtrema[i].persistence / (mRange[1] - mRange[0]);
  }

  return p;
}


std::vector<float> ExtremumGraphExt::variations() const
{
  std::vector<float> v;
  std::vector<Saddle>::const_iterator it;

  for (it=mSaddles.begin();it!=mSaddles.end();it++) {

    // If this is not a cancellation saddle
    if (!it->cancellation)
      v.push_back(it->persistence / (mRange[1] - mRange[0]));
  }

  return v;
}

std::vector<std::vector<float> > ExtremumGraphExt::extrema() const
{
  std::vector<std::vector<float> > extrema;
  for(uint32_t i=0; i<mExtrema.size(); i++) {
    std::vector<float> extremum;
    // uint32_t id;
    // float f;
    // float persistence;
    extremum.push_back(mExtrema[i].f);
    // extremum.push_back(mExtrema[i].persistence / (mRange[1] - mRange[0]));
    extremum.push_back(mExtrema[i].persistence);
    extremum.push_back(float(mExtrema[i].id));
    extrema.push_back(extremum);
  }
  return extrema;
}

float ExtremumGraphExt::criticalPointFunctionValue(uint32_t index){
  float value = 0.0;
  for(size_t i=0; i<mExtrema.size(); i++){
    if(mExtrema[i].id == index)
      return mExtrema[i].f;
  }
  for(size_t i=0; i<mSaddles.size(); i++){
    if(mSaddles[i].id == index)
      return mSaddles[i].f;
  }
  return value;
}

std::vector<float> ExtremumGraphExt::criticalPointLocation(uint32_t index){
  std::vector<float> value;

  for(size_t i=0; i<mExtrema.size(); i++){
    if(mExtrema[i].id == index)
      return mPointLocations[i];
  }
  for(size_t i=0; i<mSaddles.size(); i++){
    if(mSaddles[i].id == index)
      return mPointLocations[i];
  }
  return value;
}

uint32_t unionfind(std::vector<uint32_t>& steepest,uint32_t i)
{
  if (i == steepest[i])
    return i;

  steepest[i] = unionfind(steepest,steepest[i]);
  return steepest[i];
}

uint32_t ExtremumGraphExt::rep(uint32_t v, uint32_t count)
{
  if ((mExtrema[v].parent == v) || (v < count))
    return v;

  return rep(mExtrema[v].parent,count);
}

int32_t ExtremumGraphExt::activeExtremum(uint32_t ext, uint32_t count)
{
  std::unordered_map<uint32_t, uint32_t>::iterator it;

  it = mIndexMap.find(ext);
  if (it == mIndexMap.end()) {
    fprintf(stderr,"Error: No extremum at vertex %d\n",ext);
    return -1;
  }

  if (it->second >= count) {
    fprintf(stderr,"Error: Extremum at vertex %d is merged at count %d\n",ext,count);
    return -1;
  }

  return it->second;
}


void ExtremumGraphExt::computeSegmentation(const HDData* data, const Flags* flags,
                                           const Neighborhood* edges,
                                           const ComputeMode mode)
{
  NeighborhoodIterator it(*edges);

  computeSegmentation(data,flags,it,mode);
}

void ExtremumGraphExt::computeSegmentation(const HDData* data, const Flags* flags,
                                           EdgeIterator& eIt,
                                           const ComputeMode mode)
{
  fprintf(stderr, "---- ExtremumGraphExt::computeSegmentation ----\n");

  mSteepest.resize(data->size());
  std::vector<float> slope(data->size(),-10e34);
  uint32_t i;
  cmp smaller(data,mAscending);
  float tmp,mag;


  //! Initialize the steepest array
  for (i=0;i<data->size();i++) {
    if(flags){
        if ((*flags)[i])
          mSteepest[i] = i;
        else
          mSteepest[i] = LNULL;
    }else{
        mSteepest[i] = i;
    }
  }


  //go through each edge find the steepest for each point
  // mGradientMin = std::numeric_limits<float>::max();
  // mGradientMax = std::numeric_limits<float>::min();
  // mNeighbor.clear();
  // mNeighbor.resize(data->size());

  const uint32_t* e;
  while (!eIt.end()) {

    e = *eIt;

    //if ((e[0] == 1) || (e[1] == 1)) {
    //  fprintf(stderr,"Processing edge: {(%d,%f), (%d,%f)}\n",e[0],data->f(e[0]),e[1],data->f(e[1]));
    //  fprintf(stderr,"\t %d %d\n",mSteepest[e[0]],mSteepest[e[1]]);
    //}
    // Ignore half the edges because this is symmetric and all the ones leading
    // to invalid vertices
    if (smaller(e[1],e[0]) || (mSteepest[e[0]]==LNULL) || (mSteepest[e[1]]==LNULL)) {
      eIt++;
      continue;
    }

    if (eIt.hasLength()){
      mag = eIt.length();
    }
    else{
      mag = data->dist(e[0],e[1]);
    }

    if (mag < 1e-8)
      tmp = 10e10;
    else
      tmp = fabs(data->f(e[1]) - data->f(e[0])) / mag;

    //// store info for distance computation ///
    // float grad = (data->f(e[1]) - data->f(e[0])) / mag;
    // if(grad > mGradientMax)
    //   mGradientMax = grad;
    // if(grad < mGradientMin)
    //   mGradientMin = grad;
    // mNeighbor[e[0]].insert(std::pair<uint32_t,float>(e[1], grad));
    ////////////////////////////////////////////

    if (slope[e[0]] < tmp) {
      slope[e[0]] = tmp;
      mSteepest[e[0]] = e[1];
    }

    eIt++;
  }

  // store the per-point steepest info
  // msteepest = steepest;
  // mslope = slope;

  // Path compress all edges
  for (i=0;i<data->size();i++) {
    if (mSteepest[i] != LNULL) {
      mSteepest[i] = unionfind(mSteepest,i);

      if (mSteepest[i] == i) {
        //fprintf(stderr,"Found Extremum %d\n",i);
        mExtrema.push_back(Extremum(i,data->f(i),mExtrema.size()));
      }
    }
  }

  // A map from neighboring segments to saddles
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> saddle_edges;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator it;

  eIt.reset();

  fprintf(stderr, "---- ExtremumGraphExt::Reset EdgeIterator ----\n");

  // Find the highest lower vertex of all edges that connect two different segments
  while (!eIt.end()) {
    e = *eIt;

    if ((mSteepest[e[0]] == LNULL) || (mSteepest[e[1]] == LNULL)) {
      eIt++;
      continue;
    }

    // Make sure we sort the pair in some deterministic way
    std::pair<uint32_t,uint32_t> p(std::min(mSteepest[e[0]],mSteepest[e[1]]),
                                   std::max(mSteepest[e[0]],mSteepest[e[1]]));


    // Ignore all edges that do not cross a boundary
    if (p.first == p.second) {
      eIt++;
      continue;
    }

    it = saddle_edges.find(p);
    if (it == saddle_edges.end()) {
      if (smaller(e[0],e[1])) {
        saddle_edges[p] = e[0];
      }
      else {
        saddle_edges[p] = e[1];
      }
    }
    else {
      if (smaller(e[0],e[1])) {
        if (smaller(it->second,e[0]))
          it->second = e[0];
      }
      else {
        if (smaller(it->second,e[1]))
          it->second = e[1];
      }
    }

    eIt++;
  }

  // Assemble the index map of extrema to index
  std::map<uint32_t,uint32_t> order;
  std::map<uint32_t,uint32_t>::iterator mIt,mIt2;
  for (uint32_t i=0;i<mExtrema.size();i++)
    order[mExtrema[i].id] = i;


  // For all saddles find the new (re-ordered) index of the two neighboring
  // extrema
  for (it=saddle_edges.begin();it!=saddle_edges.end();it++) {
    mIt = order.find(it->first.first);
    mIt2 = order.find(it->first.second);

    // Create the new saddle with the right mesh-vertex id and the respective
    // extremum index
    //fprintf(stderr,"Saddle %d\t %d,%d\n",it->second,it->first.first,it->first.second);
    mSaddles.push_back(Saddle(it->second,data->f(it->second),mIt->second,mIt2->second));
  }

   // If we only need the raw complex
  if  (mode == NONE)  {
    // Free up the memory of the segmentation
    mSteepest = std::vector<uint32_t>();
  }
}

void ExtremumGraphExt::computeHierarchy()
{
  fprintf(stderr, "---- ExtremumGraphExt::computeHierarchy ----\n");

  std::priority_queue<Triple> q;
  Triple top;
  uint32_t u,v;

  // For all initial saddles compute their persistence
  for (uint32_t i=0;i<mSaddles.size();i++) {
    top.saddle = i;
    top.persistence = computePersistence(i);

    // And push them into a priority queue to sort them
    q.push(top);
  }


  while (!q.empty()) {
    // Get the guy with the currently smallest persistence
    top = q.top();
    q.pop();

    // Recompute the persistence to account for neighboring extrema
    // having been cancelled in the mean time
    top.persistence = computePersistence(top.saddle);

    // If this saddle now still has the smallest persistence attempt a cancellation
    if (top.persistence <= q.top().persistence) {

      // Get the current neigbors
      u = rep(mSaddles[top.saddle].neighbors[0]);
      v = rep(mSaddles[top.saddle].neighbors[1]);

      // A strangulation cannot be actually cancelled but we use the persistence
      // to indicate the variation of this saddle
      if (u == v) {
        mSaddles[top.saddle].persistence = top.persistence;

      }
      // If u-s is the smaller persistence we need to cancel u
      else if (fabs(mExtrema[u].f - mSaddles[top.saddle].f) <= fabs(mExtrema[v].f - mSaddles[top.saddle].f)) {
        //fprintf(stderr,"Cancelling %d\n",mExtrema[u].id);
        // Need to cancel u
        mExtrema[u].parent = v;
        mExtrema[u].persistence = top.persistence;
        mSaddles[top.saddle].persistence = top.persistence;
        mSaddles[top.saddle].cancellation = true;
      }
      else { // Need to cancel v
        //fprintf(stderr,"Cancelling %d\n",mExtrema[v].id);
        mExtrema[v].parent = u;
        mExtrema[v].persistence = top.persistence;
        mSaddles[top.saddle].persistence = top.persistence;
        mSaddles[top.saddle].cancellation = true;
      }
    }
    else // If the saddle is no longer the smallest push it back onto the queue
      q.push(top);
  }
}

void ExtremumGraphExt::computeSegments(const HDData* data)
{
  cmp smaller(data,mAscending);

  // Assemble the index map of extrema to index
  std::map<uint32_t,uint32_t> order;
  std::map<uint32_t,uint32_t>::iterator mIt,mIt2;
  for (uint32_t i=0;i<mExtrema.size();i++){
    order[mExtrema[i].id] = i;
    //fprintf(stderr,"extrema id: %d\n", mExtrema[i].id);
  }

  // Finally assemble all the segments
  mSegments.resize(mExtrema.size());

  //fprintf(stderr,"assemble all the segments %d\n", mSegments.size());
  std::vector<uint32_t>::iterator vIt;
  for (uint32_t i=0;i<mSteepest.size();i++) {

    // Simply ignore the invalid points
    if (mSteepest[i] == LNULL)
      continue;

    mIt = order.find(mSteepest[i]);

    //fprintf(stderr,"steepest %d, index: %d , %d\n", mSteepest[i], mIt->first, mIt->second);
    mSegments[mIt->second].push_back(i);

    // Make sure that the first entry of each segment is the representative extremum
    if (i == mSteepest[i])
      std::swap(mSegments[mIt->second].front(),mSegments[mIt->second].back());
  }

  // Now make sure to sort the vertices in "descending" order
  for (uint32_t i=0;i<mExtrema.size();i++) {
    std::sort(mSegments[i].rbegin(),mSegments[i].rend(),smaller);
  }

  fprintf(stderr,"finish computeSegments\n");

}


void ExtremumGraphExt::computeHistograms(const HDData* data, uint32_t cube_dim, uint32_t resolution,
                                        std::vector<HistogramType> histogramTypes, int32_t target_attr)
{
  // Example to compute all 1D and 2D Histograms for all segments

  fprintf(stderr,"computeHistograms %d\n", cube_dim);

  mDistributions.clear();

  for (uint32_t i=0;i<mExtrema.size();i++)
    mDistributions.push_back(JointDistributions(*data));

  // Adding one more joint to store the joint of whole dataset
  mDistributions.push_back(JointDistributions(*data));

  // Change dim to total number of attrs
  std::vector<std::pair<float,float> > ranges(data->attr(),
                                              std::pair<float,float>(10e34,-10e34));

  mFuncAttr = data->func();

  mSelectivity = Selectivity(mDistributions, mFuncAttr, cube_dim, resolution);

  // Change dim to total number of attrs
  // First make one pass to find all ranges
  for (uint64_t i=0;i<data->size();i++) {

    if (mSteepest[i] == LNULL)
      continue;

    for (uint32_t k=0;k<data->attr();k++) {
      ranges[k].first = std::min(ranges[k].first,(*data)[i][k]);
      ranges[k].second = std::max(ranges[k].second,(*data)[i][k]);
    }
  }
  /////////////////////////////////////////
  //// Combine for a general function to create histogram
  /////////////////////////////////////////
  // Add all 1D histograms

  for (uint32_t k=0;k<mDistributions.size();k++) {
    for (auto histogramType : histogramTypes)
      mDistributions[k].createHistogram(ranges, resolution, cube_dim, (uint32_t)histogramType, target_attr);
  }

  // ! Find which base partition to add points to, add points to that base partition
  std::map<uint32_t,uint32_t> order;
  std::map<uint32_t,uint32_t>::iterator mIt;
  for (uint32_t i=0;i<mExtrema.size();i++)
    order[mExtrema[i].id] = i;


  // Add value to joint for all
  uint32_t last_ind= mDistributions.size()-1;

  for (uint32_t i=0;i<mSteepest.size();i++) {

    if (mSteepest[i] == LNULL)
      continue;

    mIt = order.find(mSteepest[i]);
    mDistributions[mIt->second].addValue((*data)[i]);

    // ! Add value to joint histogram of the whole dataset
    mDistributions[last_ind].addValue((*data)[i]);
  }

  // std::unordered_map<uint32_t, uint32_t>::iterator mIt;
  // for (uint64_t i=0;i<data->size();i++) {
  //   mIt = mIndexMap.find(mSteepest[i]);
  //   //mDistributions[mIt->second].addValue((*data)[i]);
  //   mDistributions[rep(mIt->second,mExtrema.size())].addValue((*data)[i]);
  // }

  // ! Store 1D histogram for function value for querying the function value if in HISTOGRAM ONLY MODE

  if(mSegments.size()==0){

    // Change dim -1  to index of function

    std::string yAttr = data->attributes()[data->func()];
    mFunctionHist.clear();

    for (uint32_t i=0;i<mExtrema.size();i++)
    {
      std::vector<uint32_t> tmp;

      Histogram tmphist = mDistributions[i].get(yAttr);
      tmp.resize(pow(tmphist.resolution(),tmphist.dimension())+1,0);

      tmp[0]=mExtrema[rep(i,mExtrema.size())].id;

      for(uint32_t j = 1; j<tmp.size();j++){
        tmp[j] = tmphist.data()[j-1];
      }
      mFunctionHist.push_back(tmp);
    }
  }
  fprintf(stderr, "finish compute histogram \n");
}

std::vector<uint32_t> ExtremumGraphExt::getSelected1D(std::vector<std::string> dims, std::vector<std::vector<float> > ranges, int32_t ext, uint32_t count, int32_t targetIndex){

  std::vector<uint32_t> selectedIndex;
  uint32_t r;
  if(ext>0)
  {
    for (uint32_t i = 0;i<(uint32_t)mExtrema.size();i++) {
      // Give the parent location in terms of index for current persistent level
      r = rep(i,mSegCount);
      if(mExtrema[r].id == (uint32_t)ext)
      {
        selectedIndex.push_back(i);
      }
    }
  }
  return mSelectivity.functionQuery(dims, ranges, targetIndex, selectedIndex);
}

std::vector<uint32_t> ExtremumGraphExt::getHist(uint32_t ext, uint32_t count, std::vector<std::string> attrs, bool func,
                                                std::vector<std::string> dims, std::vector<std::vector<float> > ranges)
{
  std::vector<uint32_t> selectedIndex;
  uint32_t r;
  for (uint32_t i = 0;i<(uint32_t)mExtrema.size();i++) {
    // Give the parent location in terms of index for current persistent level
    r = rep(i,mSegCount);
    if(mExtrema[r].id == ext)
    {
      selectedIndex.push_back(i);
    }
  }


  // ! Create vector of attributes to query

  // std::vector<std::string> all_attrs = mDistributions[mFuncAttr].getAttr();
  std::vector<std::string> all_attrs = mDistributions[0].getAttr();


  std::vector<std::string> tar_attrs;

  for (uint32_t ii = 0; ii < all_attrs.size(); ii++)
  {
    if (std::find(attrs.begin(), attrs.end(), all_attrs[ii]) != attrs.end())
    {
      tar_attrs.push_back(all_attrs[ii]);
    }
    else if(std::find(dims.begin(), dims.end(), all_attrs[ii]) != dims.end())
    {
      tar_attrs.push_back(all_attrs[ii]);
    }
  }

  //! ADD PEEK FUNCTION HERE TO AVOID ASSERT FAULT THAT CAUSE SYSTEM TO CRASH
  if (mDistributions[0].peek(tar_attrs))
  {
    return mSelectivity.jointQuery(selectedIndex, attrs, func, dims, ranges);
  }
  else
  {
    return std::vector<uint32_t>();
  }

}

std::vector<uint32_t> ExtremumGraphExt::getHist(std::vector<std::string> attrs, bool func,
                                                std::vector<std::string> dims, std::vector<std::vector<float> > ranges){

  // ! Create vector of attributes to query

  std::vector<std::string> all_attrs = mDistributions[0].getAttr();

  std::vector<std::string> tar_attrs;

  for (uint32_t ii = 0; ii < all_attrs.size(); ii++)
  {
    if (std::find(attrs.begin(), attrs.end(), all_attrs[ii]) != attrs.end())
    {
      tar_attrs.push_back(all_attrs[ii]);
    }
    else if(std::find(dims.begin(), dims.end(), all_attrs[ii]) != dims.end())
    {
      tar_attrs.push_back(all_attrs[ii]);
    }
  }

  //! ADD PEEK FUNCTION HERE TO AVOID ASSERT FAULT THAT CAUSE SYSTEM TO CRASH
  if (mDistributions[0].peek(tar_attrs))
  {
    return mSelectivity.jointQuery(attrs, func, dims, ranges);
  }
  else
  {
    return std::vector<uint32_t>();
  }
}

std::vector<uint32_t> ExtremumGraphExt::getHist(std::string attr1, std::string attr2){
  //return mSelectivity.jointQuery(attrs, func, dims, ranges);
  std::vector<std::string> tmp_attrs;
  tmp_attrs.push_back(attr1);
  tmp_attrs.push_back(attr2);
  return getHist(tmp_attrs);
}


std::vector<std::vector<uint32_t> > ExtremumGraphExt::segmentHist(uint32_t count){

  if (count == mSegCount){
    return mBufferedFunctionHist;
  }

  mSegCount = std::min(count,(uint32_t)mExtrema.size());

  //vertexCmp greater(mFunction,mAscending);

  mBufferedFunctionHist.clear();

  // Get the correct segmentation
  uint32_t r;
  uint32_t pre_size;

  mBufferedFunctionHist.insert(mBufferedFunctionHist.end(),mFunctionHist.begin(),mFunctionHist.begin()+mSegCount);

  for (uint32_t i=mSegCount;i<mExtrema.size();i++) {

  // Give the parent location in terms of index for current persistent level
  r = rep(i,mSegCount);

  std::transform(mBufferedFunctionHist[r].begin(), mBufferedFunctionHist[r].end(),
                  mFunctionHist[i].begin(), mBufferedFunctionHist[r].begin(), std::plus<uint32_t>());
  }


  for (uint32_t i=0;i<mSegCount;i++) {
    r = rep(i,mSegCount);
    if (mBufferedFunctionHist[i][0] != mExtrema[r].id) {
      mBufferedFunctionHist[i][0] = mExtrema[r].id;
    }
  }
  return mBufferedFunctionHist;
}

uint32_t ExtremumGraphExt::histogramSize(uint32_t ext, uint32_t count, float threshold)
{
  int32_t v = activeExtremum(ext,count);

  if (v < 0)
    return 0;

  segmentation(count);

  //! segment size for histogram only mode

  uint32_t histsize = 0;

  std::vector<uint32_t> hist = mBufferedFunctionHist[v];

  uint32_t thresInd = (uint32_t)((hist.size()-1)*(threshold-mRange[0])/(mRange[1]-mRange[0])+1);


  if(mAscending){
    for(size_t index = thresInd; index < hist.size(); index++) {
      histsize += hist[index];
    }
  }
  else{
    for(size_t index = 1; index < thresInd; index++) {
      histsize += hist[index];
    }
  }

  return histsize;
}

std::vector<uint32_t> ExtremumGraphExt::histogram(uint32_t ext, uint32_t count, float threshold)
{
  int32_t v = activeExtremum(ext,count);

  if (v < 0)
    return std::vector<uint32_t>();

  // Make sure we compute the segmentation for the right count
  segmentation(count);

  std::vector<uint32_t> partial;
  std::vector<uint32_t> hist = mBufferedFunctionHist[v];

  uint32_t thresInd = (uint32_t)((threshold-mRange[0])/(mRange[1]-mRange[0])*(hist.size()-1));

  uint32_t histsize = histogramSize(ext,count,threshold);
  //partial.push_back(ext);

  if(mAscending){
    //partial.push_back(thresInd);
    partial.push_back(histsize);
    //partial.push_back(hist.size());
  }
  else{
    //partial.push_back(1);
    partial.push_back(histsize);
    //partial.push_back(thresInd-1);
  }

  return partial;
}
void ExtremumGraphExt::simplify(uint32_t count)
{
  fprintf(stderr, "---- ExtremumGraphExt::simplify ----\n");

  if (count >= mExtrema.size())
    return;

  uint32_t i;

  // For all the extrema that will disappear
  for (uint32_t i=count;i<mExtrema.size();i++) {
    // Re-route the maximum to its new parent
    mSteepest[mExtrema[i].id] = mExtrema[rep(i,count)].id;
  }

  // Now do another union-find to fix the segmentation accordingly
  for (uint32_t i=0;i<mSteepest.size();i++) {
    if (mSteepest[i] != LNULL)
      mSteepest[i] = unionfind(mSteepest,i);
  }

  // Now go through all saddles and remove the ones that are no longer saddles
  // Note that this means we are throwing away loops
  i = 0;
  while (i<mSaddles.size()) {
    // IF this is not longer a saddle
    if (rep(mSaddles[i].neighbors[0],count) == rep(mSaddles[i].neighbors[1],count)) {
      std::swap(mSaddles[i],mSaddles.back()); // remove it
      mSaddles.pop_back();
    }
    else{
      // Replace the Neighbor with New extrema
      mSaddles[i].neighbors[0] = rep(mSaddles[i].neighbors[0],count);
      mSaddles[i].neighbors[1] = rep(mSaddles[i].neighbors[1],count);
      i++;
    }
  }

  // Re-sort the saddles again
  std::sort(mSaddles.rbegin(), mSaddles.rend(),saddleCmp());

  // And remove all extrema that are no longer valid
  mExtrema.resize(count);

}


void ExtremumGraphExt::sort(const HDData* data)
{
  //fprintf(stderr, "             SORT   \n");
  std::vector<uint32_t> order(mExtrema.size());
  std::vector<uint32_t> map(mExtrema.size());

  for (uint32_t i=0;i<mExtrema.size();i++)
    order[i] = i;

  // Sort all extrema by descending persistence
  ExtremaCmp cmp(mExtrema);
  std::sort(order.rbegin(),order.rend(),cmp);

  // Now assemble the reverse map
  for (uint32_t i=0;i<mExtrema.size();i++) {
    map[order[i]] = i;
  }

  // Now make an ordered copy of the extrema and fix the parent pointers
  std::vector<Extremum> new_extrema(mExtrema.size());
  std::vector<std::vector<uint32_t> > new_segments(mExtrema.size());

  for (uint32_t i=0;i<mExtrema.size();i++) {
    new_extrema[i] = mExtrema[order[i]];
    new_extrema[i].parent = map[new_extrema[i].parent];

    if (!mSegments.empty())
      new_segments[i] = mSegments[order[i]];

    mIndexMap[new_extrema[i].id] = i;

  }


  // Swap with the new data
  mExtrema = new_extrema;
  mSegments = new_segments;

  // Finally, fix the neighbor pointer of all saddles
  for (uint32_t i=0;i<mSaddles.size();i++) {
    mSaddles[i].neighbors[0] = map[mSaddles[i].neighbors[0]];
    mSaddles[i].neighbors[1] = map[mSaddles[i].neighbors[1]];
  }

  // And sort all the saddles
  std::sort(mSaddles.rbegin(), mSaddles.rend(),saddleCmp());

}

void ExtremumGraphExt::storeLocations(const HDData* data){

  fprintf(stderr, "---- ExtremumGraphExt::storeLocations ----\n");

  // And point locations for remaining critical pts
  for(size_t i=0; i<mExtrema.size(); i++){
    // change dim-1 to dim
    std::vector<float> locs((*data)[mExtrema[i].id], (*data)[mExtrema[i].id] + data->dim());
    mPointLocations.push_back(locs);

  }
  for(size_t i=0; i<mSaddles.size(); i++){
    // change dim-1 to dim
    std::vector<float> locs((*data)[mSaddles[i].id], (*data)[mSaddles[i].id] + data->dim());
    mPointLocations.push_back(locs);
  }
}

float ExtremumGraphExt::computePersistence(uint32_t saddle)
{
  uint32_t left = rep(mSaddles[saddle].neighbors[0]);
  uint32_t right = rep(mSaddles[saddle].neighbors[1]);

  return std::min(fabs(mSaddles[saddle].f - mExtrema[left].f),
                  fabs(mSaddles[saddle].f - mExtrema[right].f));
}

std::vector<uint32_t> ExtremumGraphExt::activeGraph(uint32_t count, float variation)
{

  std::vector<uint32_t> saddles;
  uint32_t left,right;

  variation *= (mRange[1] - mRange[0]);


  for (uint32_t i=0;i<mSaddles.size();i++) {
    // First find the active extremum on either side
    left = rep(mSaddles[i].neighbors[0],count);
    right = rep(mSaddles[i].neighbors[1],count);

    // We do not want to show trivial loops
    if (left == right)
      continue;

    // If the saddle is part of the cancellation tree or has a small variation threshold
    if (mSaddles[i].cancellation || (mSaddles[i].persistence < variation)) {
      saddles.push_back(mSaddles[i].id);
      saddles.push_back(mExtrema[left].id);
      saddles.push_back(mExtrema[right].id);
    }
  }

  // When there is no active saddle ?
  // if(saddles.empty()){
  //   float tmp = mExtrema[0].f;
  //   uint32_t tmp_id = 0;
  //   for (uint32_t i=0;i<mExtrema.size();i++) {
  //     if(mExtrema[i].f>=tmp)
  //       tmp_id = mExtrema[i].id;
  //   }
  //   saddles.push_back(tmp_id);
  //   saddles.push_back(tmp_id);
  //   saddles.push_back(tmp_id);
  // }
  return saddles;
}

///////////////// IO ///////////////////
bool ExtremumGraphExt::load(HDFileFormat::DataBlockHandle &handle, bool isIncludeFunctionIndexInfo, uint32_t cube_dim){

    //load saddle
    HDFileFormat::DataBlockHandle* saddles = handle.getChildByType<HDFileFormat::DataBlockHandle>("Saddles");
    //hderror(saddles==NULL,"No saddle found");
    mSaddles.resize(saddles->sampleCount());
    saddles->readData(&mSaddles[0]);

    //load extrema
    HDFileFormat::DataBlockHandle* extrema = handle.getChildByType<HDFileFormat::DataBlockHandle>("Extrema");
    //hderror(extrema==NULL,"No saddle found");
    mExtrema.resize(extrema->sampleCount());
    extrema->readData(&mExtrema[0]);

    /////// de-serialize ////////
    HDFileFormat::DataBlockHandle *bufferHandle = handle.getChildByType<HDFileFormat::DataBlockHandle>("serializedMetaData");
    //hderror(bufferHandle==NULL,"No serializedMetaData found");
    mSerializationBuffer.clear();
    mSerializationBuffer.resize(bufferHandle->sampleCount());
    bufferHandle->readData(&mSerializationBuffer[0]);

    // for(size_t i=0; i<mSerializationBuffer.size(); i++)
    //   std::cout<<mSerializationBuffer[i];
    // std::cout<<std::endl;

    // membuf inBuffer(&mSerializationBuffer[0], mSerializationBuffer.size(), false);
    autoResizeMembuf inBuffer(false);
    std::istream is(&inBuffer);
    inBuffer.setBuffer(mSerializationBuffer);
    this->deserialize(is, isIncludeFunctionIndexInfo);


    /////////////////////////////
    // std::cout << "mIndexMap:";
    // for ( auto it = mIndexMap.begin(); it != mIndexMap.end(); ++it )
    //   std::cout << " " << it->first << ":" << it->second;
    // std::cout << std::endl;

    //load all distribution
    //The list of distrbution handle (hold access to histograms)
    std::vector<HDFileFormat::DistributionHandle> distHandleList;
    handle.getAllChildrenByType<HDFileFormat::DistributionHandle>(distHandleList);
    fprintf(stderr, "find %ld JointDistribution\n", distHandleList.size());
    mDistributions.resize(distHandleList.size());
    for(size_t i=0; i<distHandleList.size(); i++){
      mDistributions[i].load(distHandleList[i]);
    }

    if(distHandleList.size()!=0){
        if(!isIncludeFunctionIndexInfo)
          mFuncAttr = mDistributions[0].getAttr().size()-1;

        mCubeDim = cube_dim;

        // Find the highest resolution for the data cube
        uint32_t resolution = mDistributions[0].get(mDistributions[0].getAttr()[mFuncAttr]).resolution();

        std::cout<< "Highest Res in the cube = "<< resolution<<"\n";
        mSelectivity = Selectivity(mDistributions, mFuncAttr, mCubeDim, resolution);
    }

    return true;
}

bool ExtremumGraphExt::save(HDFileFormat::DataBlockHandle &handle){

    HDFileFormat::DataBlockHandle blockSal;
    blockSal.idString("Saddles");
    blockSal.setData<Saddle>(&mSaddles[0],mSaddles.size());
    handle.add(blockSal);

    HDFileFormat::DataBlockHandle blockExt;
    blockExt.idString("Extrema");
    blockExt.setData<Extremum>(&mExtrema[0],mExtrema.size());
    handle.add(blockExt);

    ////////// serialize /////////
    // mSerializationBuffer.resize(102400000);
    // membuf streamBuffer(&mSerializationBuffer[0], mSerializationBuffer.size());
    autoResizeMembuf streamBuffer;
    std::ostream os(&streamBuffer);
    // fprintf(stderr, " ---- before serialize\n");
    this->serialize(os);
    // fprintf(stderr, " ---- after serialize\n");
    streamBuffer.copyBuffer(mSerializationBuffer);
    // for(size_t i=0; i<streamBuffer.outputCount(); i++)
    //   std::cout<<mSerializationBuffer[i];
    // std::cout<<std::endl;

    // std::cout << "mIndexMap:";
    // for ( auto it = mIndexMap.begin(); it != mIndexMap.end(); ++it )
    //   std::cout << " " << it->first << ":" << it->second;
    // std::cout << std::endl;

    // fprintf(stderr, " == done serialization == %ld\n", streamBuffer.outputCount());
    HDFileFormat::DataBlockHandle bufferHandle;
    bufferHandle.idString("serializedMetaData");
    bufferHandle.setData(&mSerializationBuffer[0], streamBuffer.outputCount());
    handle.add(bufferHandle);
    ///////////////////////////////

    // /*
    fprintf(stderr, "mDistributions size: %ld\n", mDistributions.size());
    for(size_t i=0; i<mDistributions.size(); i++){
      HDFileFormat::DistributionHandle distHandle;
      distHandle.idString("distribution");
      mDistributions[i].save(distHandle);
      handle.add(distHandle);
    }
    // */


    return true;
}


void ExtremumGraphExt::serialize(std::ostream &output){
  cereal::BinaryOutputArchive binAR(output);
  // cereal::XMLOutputArchive binAR(output);
  binAR( this->mIndexMap, this->mAscending, this->mRange,this->mFunction, this->mSegments, this->mSegmentation, this->mFunctionHist, this->mPointLocations, this->mFuncAttr);
}

void ExtremumGraphExt::deserialize(std::istream &input, bool isIncludeFunctionIndexInfo){
  cereal::BinaryInputArchive binAR(input);
  // cereal::XMLInputArchive binAR(input);
  if(isIncludeFunctionIndexInfo)
    binAR( this->mIndexMap, this->mAscending, this->mRange, this->mFunction, this->mSegments, this->mSegmentation, this->mFunctionHist, this->mPointLocations, this->mFuncAttr);
  else
  {
    binAR( this->mIndexMap, this->mAscending, this->mRange, this->mFunction, this->mSegments, this->mSegmentation, this->mFunctionHist, this->mPointLocations);
  }
}
