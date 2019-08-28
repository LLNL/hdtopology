/*
 * JointDistributions.h
 *
 *  Created on: Oct 3, 2018
 *      Author: bremer5
 */

#ifndef JOINTDISTRIBUTIONS_H
#define JOINTDISTRIBUTIONS_H

#include <vector>
#include <string>
#include <unordered_map>
#include <stdint.h>
#include <math.h>
#include "Histogram.h"

//for writting to file
#include <DistributionHandle.h>

//! Class to hold joint histograms from up to 255 dimensions
//! and up to 8 way histograms
class JointDistributions
{
public:

  //! Histogram Mode
  enum HistogramType {
    REGULAR = 0,
    REDUCED = 1,
    ENTROPY = 2,
    DTREE = 3
  };

  //! Default constructor
  JointDistributions(){}

  //! Data constructor
  JointDistributions(const HDData& data);

  //! Set HDData
  void SetData(const HDData& data);

  //! Add the given histogram to the set
  int addHistogram(std::vector<std::string>& attributes,
                   std::vector<std::pair<float,float> >& ranges,
                   uint32_t resolution);

  //! Add the given histogram to the set
  int addHistogram(std::vector<uint8_t>& indices,
                   std::vector<std::pair<float,float> >& ranges,
                   uint32_t resolution);

  //! A general function to create n-d histogram in specified mode
  void createHistogram(std::vector<std::pair<float,float> >& ranges,
                        uint32_t resolution, uint32_t cubeDim, uint32_t histogramType,  int32_t target_attr);//HistogramType histogramType);

  // //! A general function to create n-d histogram in specified mode
  // void createHistogram(std::vector<std::string>& attributes,
  //                  std::vector<std::pair<float,float> >& ranges,
  //                  uint32_t resolution, uint32_t cubeDim, HistogramType histogramType);

  //! Get a specific Histogram
  const Histogram& get(std::string attr1);

  //! Get a specific Histogram
  const Histogram& get(std::string attr1, std::string attr2);

  //! Get a specific Histogram
  const Histogram& get(std::string attr1, std::string attr2, std::string attr3);

  //! Peek a joint Distribution
  bool peek(const std::vector<std::string>& attributes);

  //! Get attrs, will be removed later
  std::vector<std::string> getAttr() const {return mAttributes;}

  //! Compute all histograms
  void computeHistograms(const HDData& data);

  //! Add individual samples to all Histograms
  void addValue(const float* sample);

  //file io
  bool load(HDFileFormat::DistributionHandle &handle);
  bool save(HDFileFormat::DistributionHandle &handle);

  //serialization metaInfo
  void serialize(std::ostream &output);
  void deserialize(std::istream &input);

private:

  void multiResolutionHistogram(std::vector<std::pair<float,float> >& ranges,
                                std::vector<uint32_t>& resolutions, int32_t target_attr);
  //! The set of all histograms
  std::vector<Histogram> mHistograms;

  //! List of names
  std::vector<std::string> mAttributes;

  //! The map of names to indices
  std::unordered_map<std::string,uint8_t> mNameMap;

  //! Maps the to the various histograms
  std::unordered_map<uint64_t,uint32_t> mHistogramMap;

  //! Get a specific Histogram
  const Histogram& get(const std::vector<std::string>& attributes);

  //! Get a specific Histogram
  const Histogram& get(const std::vector<uint8_t>& indices);

  //! serialization buffer
  std::vector<char> mSerializationBuffer;

};



#endif
