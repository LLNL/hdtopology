/*
 * Histogram.h
 *
 *  Created on: Sep 25, 2018
 *      Author: bremer5
 */

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <tuple>
#include <stdint.h>
#include <cmath>
#include <cassert>
#include "HDData.h"
#include <numeric>

#include <HistogramHandle.h>

class Histogram
{
public:
  //! empty constructor
  Histogram(){}


  //! Convinience typedef to allow fast addValue function
  typedef void(Histogram::*HistMemFn)(const float* sample);

  //! Default constructor
  Histogram(std::vector<std::string> attributes, std::vector<uint8_t> indices,
            std::vector<std::pair<float,float> > &ranges, uint32_t resolution);

  //Histogram(const Histogram& hist):mDimension(hist.dimension()), mResolution(hist.resolution()),mRange(hist.range())
  //{
  //}

  //! Return the number of dimension
  uint8_t dimension() const {return mDimension;}

  //! Return the resolution used
  uint32_t resolution() const {return mResolution;}


  // std::vector<std::pair<float,float> > range() const {return mRange;}

  //! Return a pointer to the internal data array
  const uint32_t* data() const {return &mData[0];}

  // std::vector<std::pair<float,float> > ranges()const {return mRange;}
  std::vector<std::pair<float,float> > ranges() const;

  //! Return the name of the i'th attribute
  std::string attribute(uint8_t i) const {return mNames[i];}

  //! Set the ranges
  //void setRange(std::vector<std::pair<float,float> >& range) {mRange = range;}

  //! Main call for histogram computation
  void addValue(const float* sample) {(this->*mAddValueCalls[mDimension])(sample);}

  //IO
  bool load(HDFileFormat::HistogramHandle &handle);
  bool save(HDFileFormat::HistogramHandle &handle);
   //serialization metaInfo
  void serialize(std::ostream &output);
  void deserialize(std::istream &input);

  //Histogram& operator+(const Histogram& hist);
  //! Add Histogram data to a class, not sure whether this is the best way.
  void data(std::vector<uint32_t> &hist){mData = hist;}

  //! Addition operator for Histogram
  Histogram operator+(const Histogram& hist);

  //! Copy constructor
  Histogram& operator=(const Histogram & hist);

private:

  //! The dimension of the histogram for simplifity
  // const uint8_t mDimension;
  // can not be const if we want to load from file
  uint8_t mDimension;

  //! The resolution of the histogram
  // const uint32_t mResolution;
  uint32_t mResolution;

  //! The ranges of my dimension
  //const
  std::vector<std::tuple<float,float> > mRange;
  // std::vector<std::pair<float,float> > mRange;

  //! The attribute ids
  std::vector<uint8_t> mIndices;

  //! The labels of the dimension for convinience
  std::vector<std::string> mNames;

  //! The counts of the histogram
  std::vector<uint32_t> mData;

  //! Pointers to addValue functions to optimize computation
  std::vector<HistMemFn> mAddValueCalls;

  //! Specialized function for 1D histograms
  void addValue1D(const float* sample);

  //! Specialized function for 2D histograms
  void addValue2D(const float* sample);

  //! Specialized function for 3D histograms
  void addValue3D(const float* sample);

  //! Specialized function for 4D histograms
  void addValue4D(const float* sample);

  //serialization buffer
  std::vector<char> mSerializationBuffer;
};




#endif
