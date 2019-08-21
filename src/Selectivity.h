#ifndef SELECTIVITY_H
#define SELECTIVITY_H

#include <cstdint>

#include <vector>
#include <string>
#include <unordered_map>
#include <stdint.h>
#include "JointDistributions.h"
#include "Histogram.h"
//for writting to file
// #include <HDFileFormat/DistributionHandle.h>

class Selectivity
{
public:

  //! Default constructor
  Selectivity(){}

  Selectivity(std::vector<JointDistributions>& distributions, uint32_t funcAttr, uint32_t cubeDim, uint32_t targetRes);

  std::vector<uint32_t> jointQuery(std::string attr1, std::string attr2);
  std::vector<uint32_t> jointQuery(std::string attr1, std::string attr2, std::string attr3);
  std::vector<uint32_t> jointQuery(std::vector<std::string>& attrs, bool func=false, 
                                   std::vector<std::string> dims=std::vector<std::string>(), 
                                   std::vector<std::vector<float> > ranges=std::vector<std::vector<float> >());

  std::vector<uint32_t> jointQuery(std::vector<uint32_t>& selectedIndex, std::vector<std::string>& attrs, bool func=false, 
                                   std::vector<std::string> dims=std::vector<std::string>(), 
                                   std::vector<std::vector<float> > ranges=std::vector<std::vector<float> >());
  std::vector<uint32_t> jointQuery(std::vector<uint32_t>& selectedIndex, std::string attr1, std::string attr2);
  std::vector<uint32_t> jointQuery(std::vector<uint32_t>& selectedIndex, std::string attr1, std::string attr2, std::string attr3);

  std::vector<uint32_t> functionQuery(std::vector<std::string>& dims, std::vector<std::vector<float> >& ranges, int32_t targetIndex=-1, 
                                      std::vector<uint32_t> selectedIndex=std::vector<uint32_t>());



private:

  std::vector<uint32_t> interpolateHist(std::vector<uint32_t>& inputHist, uint32_t inputRes, uint32_t outputRes);
  std::vector<uint32_t> interpolateHist(std::vector<uint32_t>& inputHist, std::vector<uint32_t>& targetHist, uint32_t inputRes, uint32_t outputRes);
//   std::vector<uint32_t> cubeQuery(std::vector<uint32_t>& inputCube, uint32_t cubeDim, uint32_t res, 
//                                   std::vector<std::vector<float> >ranges, std::vector<uint32_t> outDims);

  std::vector<JointDistributions> mDistributions;
  uint32_t mFuncAttr;
  uint32_t mSize;
  std::vector<std::string>  mAttrs;
  uint32_t mCubeDim;
  //! Highest Resolution for the data cube 
  uint32_t mTargetRes;

};



#endif
