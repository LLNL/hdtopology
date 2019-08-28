/*
 * Histogram.cpp
 *
 *  Created on: Oct 3, 2018
 *      Author: bremer5
 */

#include <algorithm>
#include "Histogram.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
//#include <cereal/types/pair.hpp> !! No pair support
#include <cereal/types/tuple.hpp>
#include <membuf.h>

Histogram::Histogram(std::vector<std::string> attributes, std::vector<uint8_t> indices,
                     std::vector<std::pair<float,float> >& ranges, uint32_t resolution) :
                     mDimension(attributes.size()), mResolution(resolution),
                     mIndices(indices), mNames(attributes)
                    //  ,mRange(ranges)
{
  //convert from pair to tuple
  for(size_t i=0; i<ranges.size(); i++)
    mRange.push_back(std::make_tuple(ranges[i].first, ranges[i].second));

  mData.resize(pow(resolution,attributes.size()),0);
  mAddValueCalls.resize(5,NULL);
  mAddValueCalls[1] = &Histogram::addValue1D;
  mAddValueCalls[2] = &Histogram::addValue2D;
  mAddValueCalls[3] = &Histogram::addValue3D;
  mAddValueCalls[4] = &Histogram::addValue4D;
}

Histogram& Histogram::operator=(const Histogram & hist)
{
  mDimension = hist.mDimension;
  mResolution = hist.mResolution;
  mRange = hist.mRange;

  //FIXME use std copy instead
  std::vector<uint32_t> tmp;
  tmp.resize(pow(resolution(),dimension()),0);
  for(uint32_t ii = 0;ii<tmp.size();ii++)
    tmp[ii]=hist.data()[ii];

  data(tmp);

  return *this;
}

std::vector<std::pair<float,float> > Histogram::ranges() const {
  //convert from tuple to pair
  std::vector<std::pair<float,float> > range;
  for(size_t i=0; i<mRange.size(); i++)
  {
    range.push_back(std::pair<float, float>( std::get<0>(mRange[i]), std::get<1>(mRange[i])));
  }
  return range;
}

void Histogram::addValue1D(const float* sample)
{
  static uint32_t bin;

  // bin = std::min(mResolution-1,std::max((uint32_t)0,
  //                                       (uint32_t)(mResolution*(sample[mIndices[0]] - mRange[0].first)
  //                                           / (mRange[0].second - mRange[0].first))));
  bin = std::min(mResolution-1,std::max((uint32_t)0,
                                        (uint32_t)(mResolution*(sample[mIndices[0]]
                                          - std::get<0>(mRange[0]))
                                            / (std::get<1>(mRange[0]) - std::get<0>(mRange[0]) ))));
  mData[bin]++;
}

void Histogram::addValue2D(const float* sample)
{
  static uint32_t bin[2];
  // bin[0] = std::min(mResolution-1,std::max((uint32_t)0,
  //                                          (uint32_t)(mResolution*(sample[mIndices[0]] - mRange[0].first)
  //                                              / (mRange[0].second - mRange[0].first))));
  // bin[1] = std::min(mResolution-1,std::max((uint32_t)0,
  //                                          (uint32_t)(mResolution*(sample[mIndices[1]] - mRange[1].first)
  //                                              / (mRange[1].second - mRange[1].first))));
  // bin[2] = std::min(mResolution-1,std::max((uint32_t)0,
  //                                          (uint32_t)(mResolution*(sample[mIndices[2]] - mRange[2].first)
  //                                              / (mRange[2].second - mRange[2].first))));
  bin[0] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[0]] - std::get<0>(mRange[0]))
                                               / (std::get<1>(mRange[0]) - std::get<0>(mRange[0])))));
  bin[1] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[1]] - std::get<0>(mRange[1]))
                                               / (std::get<1>(mRange[1]) - std::get<0>(mRange[1])))));
  mData[bin[0]*mResolution + bin[1]]++;
  //mData[bin[1]*mResolution + bin[0]]++;

}

void Histogram::addValue3D(const float* sample)
{
  static uint32_t bin[3];

  bin[0] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[0]] - std::get<0>(mRange[0]))
                                               / (std::get<1>(mRange[0]) - std::get<0>(mRange[0])))));
  bin[1] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[1]] - std::get<0>(mRange[1]))
                                               / (std::get<1>(mRange[1]) - std::get<0>(mRange[1])))));
  bin[2] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[2]] - std::get<0>(mRange[2]))
                                               / (std::get<1>(mRange[2]) - std::get<0>(mRange[2])))));
  mData[(bin[0]*mResolution + bin[1])*mResolution + bin[2]]++;
  //mData[(bin[2]*mResolution + bin[1])*mResolution + bin[0]]++;

}


void Histogram::addValue4D(const float* sample)
{
  static uint32_t bin[4];

  bin[0] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[0]] - std::get<0>(mRange[0]))
                                               / (std::get<1>(mRange[0]) - std::get<0>(mRange[0])))));
  bin[1] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[1]] - std::get<0>(mRange[1]))
                                               / (std::get<1>(mRange[1]) - std::get<0>(mRange[1])))));
  bin[2] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[2]] - std::get<0>(mRange[2]))
                                               / (std::get<1>(mRange[2]) - std::get<0>(mRange[2])))));
  bin[3] = std::min(mResolution-1,std::max((uint32_t)0,
                                           (uint32_t)(mResolution*(sample[mIndices[3]] - std::get<0>(mRange[3]))
                                               / (std::get<1>(mRange[3]) - std::get<0>(mRange[3])))));

  mData[((bin[0]*mResolution + bin[1])*mResolution + bin[2])*mResolution+bin[3]]++;
  //mData[(bin[2]*mResolution + bin[1])*mResolution + bin[0]]++;

}


Histogram Histogram::operator+(const Histogram& hist)
{
  assert(dimension() == hist.dimension());
  assert(resolution() == hist.resolution());

  // Need to use std::vector<std::string> attributes,
  //                std::vector<uint8_t> indices,
  //                std::vector<std::pair<float,float> >& ranges
  // For add operator
  // Not sure whether this is the best way

  // std::vector<std::pair<float,float> > new_range = mRange;
  std::vector<std::pair<float,float> > new_range;
  for(size_t i=0; i<mRange.size(); i++)
    new_range.push_back(std::pair<float, float>(std::get<0>(mRange[i]),
                                                std::get<1>(mRange[i]) ));

  Histogram output = Histogram(mNames, mIndices, new_range, resolution());

  std::vector<uint32_t> temp_hist = mData;

  for(uint32_t ii = 0;ii<temp_hist.size();ii++){
    temp_hist[ii]+=hist.data()[ii];
  }

  output.data(temp_hist);

  return output;
}

/////////////// IO ///////////////
bool Histogram::load(HDFileFormat::HistogramHandle &handle){
  mData.resize(handle.sampleCount());
  handle.readData(&mData[0]);

  //load class states
  HDFileFormat::DataBlockHandle *bufferHandle = handle.getChildByType<HDFileFormat::DataBlockHandle>("serializedMetaData");
  //de-serialize
  mSerializationBuffer.clear();
  mSerializationBuffer.resize(bufferHandle->sampleCount());
  bufferHandle->readData(&mSerializationBuffer[0]);

  // membuf streamBuffer(&mSerializationBuffer[0], mSerializationBuffer.size(), false);
  autoResizeMembuf inBuffer(false);
  inBuffer.setBuffer(mSerializationBuffer);
  std::istream is(&inBuffer);
  this->deserialize(is);
  return true;
}

bool Histogram::save(HDFileFormat::HistogramHandle &handle){

  handle.idString("HistoData");
  // fprintf(stderr, "mData size: %ld\n", mData.size());
  handle.setData<uint32_t>(&mData[0], mData.size());

  // meta data
  //FIXME non-resizable buffer size
  // mSerializationBuffer.resize(1024000);
  autoResizeMembuf streamBuffer;
  // membuf streamBuffer(&mSerializationBuffer[0], mSerializationBuffer.size());
  // {
  std::ostream os(&streamBuffer);
  this->serialize(os);
  // }
  streamBuffer.copyBuffer(mSerializationBuffer);

  HDFileFormat::DataBlockHandle bufferHandle;
  bufferHandle.idString("serializedMetaData");
  bufferHandle.setData(&mSerializationBuffer[0], streamBuffer.outputCount());
  handle.add(bufferHandle);
  return true;
}

void Histogram::serialize(std::ostream &output)
{
  // Meta Info for serialization
  // uint8_t mDimension;
  // uint32_t mResolution;
  // const std::vector<std::pair<float,float> > mRange;
  // std::vector<uint8_t> mIndices;
  // std::vector<std::string> mNames;

  cereal::BinaryOutputArchive binAR(output);
  //remove this->mRange since cereal do not have support for std::pair
  binAR(this->mRange, this->mDimension, this->mResolution, this->mIndices, this->mNames );

}

void Histogram::deserialize(std::istream &input){
  cereal::BinaryInputArchive binAR(input);
  binAR(this->mRange, this->mDimension, this->mResolution, this->mIndices, this->mNames );
}
