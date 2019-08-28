/*
 * JointDistribution.cpp
 *
 *  Created on: Oct 3, 2018
 *      Author: bremer5
 */

#include "JointDistributions.h"
#include <HistogramHandle.h>
#include <cereal/archives/binary.hpp>
// #include <cereal/archives/xml.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <membuf.h>
#include <iostream>


JointDistributions::JointDistributions(const HDData& data)
{
  this->SetData(data);
}

void JointDistributions::SetData(const HDData& data){
  mAttributes = data.attributes();

  // Assemble the name map

  // change name to attr
  for (uint8_t i=0;i<data.attr();i++){
    mNameMap[data.attribute(i)] = i;
  }

}

int JointDistributions::addHistogram(std::vector<std::string>& attributes,
                                     std::vector<std::pair<float,float> >& ranges,
                                     uint32_t resolution)
{
  std::vector<uint8_t> indices(attributes.size());
  std::unordered_map<std::string,uint8_t>::iterator mIt;
  uint64_t code=0;

  for (uint8_t i=0;i<indices.size();i++) {
    mIt = mNameMap.find(attributes[i]);
    if (mIt == mNameMap.end()) {
      fprintf(stderr,"Could not find attribute %s .. did not add histogram\n",
              attributes[i].c_str());
      return 1;
    }
    else {
      indices[i] = mIt->second;
      code += indices[i] << 4*i;//sizeof(uint8_t)*i;
    }
  }

  mHistograms.push_back(Histogram(attributes,indices,ranges,resolution));

  // Now we make sure that we can find this histogram
  mHistogramMap[code] = mHistograms.size()-1;

  return 0;
}

int JointDistributions::addHistogram(std::vector<uint8_t>& indices,
                                     std::vector<std::pair<float,float> >& ranges,
                                     uint32_t resolution)
{
  std::vector<std::string> attributes(indices.size());
  uint64_t code=0;

  for (uint8_t i=0;i<indices.size();i++) {
    if (indices[i] >= mAttributes.size()) {
      fprintf(stderr,"Could not find index %d .. did not add histogram\n",
              indices[i]);
      return 1;
    }
    else {
      attributes[i] = mAttributes[indices[i]];
      code += (uint64_t)indices[i] << 4*i;//sizeof(uint8_t)*i;

    }

  }

  mHistograms.push_back(Histogram(attributes,indices,ranges,resolution));
  // Now we make sure that we can find this histogram
  mHistogramMap[code] = mHistograms.size()-1;

  return 0;
}

void JointDistributions::createHistogram( std::vector<std::pair<float,float> >& ranges,
                                          uint32_t resolution, uint32_t cubeDim, uint32_t histogramType, int32_t target_attr)//HistogramType histogramType)
{
  //! Right now the Reduced Histogram's resolution decreases by a factor of 2 with increasing dimension for the cube
  if(histogramType == REGULAR)
  {
    std::vector<uint32_t> resolutions (cubeDim, resolution);
    multiResolutionHistogram(ranges, resolutions, target_attr);
  }
  else if(histogramType == REDUCED)
  {
    std::vector<uint32_t> resolutions (cubeDim, resolution);
    for (uint32_t i = 0; i<cubeDim; i++){
      if(i>=2)
        resolutions[i] = resolution/((uint32_t)pow(2, i-1));
    }
    multiResolutionHistogram(ranges, resolutions, target_attr);
  }
}

void JointDistributions::multiResolutionHistogram(std::vector<std::pair<float,float> >& ranges,
                                                  std::vector<uint32_t>& resolutions, int32_t target_attr)
{

  // Add all 1D histograms
  std::vector<uint8_t> indices(1);
  std::vector<std::pair<float,float> > range(1);

  for (uint8_t i=0;i<ranges.size();i++) {
    indices[0] = i;
    range[0] = ranges[i];

    addHistogram(indices, range, resolutions[0]);
  }


  // Add all 2D histograms
  indices.resize(2);
  range.resize(2);
  // Change dim to total number of attrs

  for (uint8_t i=0;i<ranges.size();i++) {
    indices[0] = i;
    range[0] = ranges[i];

    // Change dim to total number of attrs
    for (uint8_t j=i+1;j<ranges.size();j++) {
      indices[1] = j;
      range[1] = ranges[j];

      addHistogram(indices, range, resolutions[1]);
    }
  }

  // Add all 3D histograms, may be used later
  // ???

  if((resolutions.size()>2)&&(ranges.size()>2))
  {
    indices.resize(3);
    range.resize(3);
    for (uint8_t i=0;i<ranges.size();i++) {
      indices[0] = i;
      range[0] = ranges[i];
      for (uint8_t j=i+1;j<ranges.size();j++) {
        indices[1] = j;
        range[1] = ranges[j];
        for(uint8_t k=j+1;k<ranges.size();k++){
          indices[2] = k;
          range[2] = ranges[k];
          // ! This might be rewritten for speed issue
          if((target_attr==-1)||(((uint8_t)target_attr)==i)||(((uint8_t)target_attr)==j)||(((uint8_t)target_attr)==k) )
            addHistogram(indices, range, resolutions[2]);
        }
      }
    }
  }

  // 4D Data Cube
  if((resolutions.size()>3)&&(ranges.size()>3))
  {
    indices.resize(4);
    range.resize(4);
    for (uint8_t i=0;i<ranges.size();i++) {
      indices[0] = i;
      range[0] = ranges[i];
      for (uint8_t j=i+1;j<ranges.size();j++){
        indices[1] = j;
        range[1] = ranges[j];
        for(uint8_t k=j+1;k<ranges.size();k++){
          indices[2] = k;
          range[2] = ranges[k];
          for(uint8_t l=k+1;l<ranges.size();l++){
            indices[3] = l;
            range[3] = ranges[l];

            addHistogram(indices, range, resolutions[3]);
          }
        }
      }
    }
  }
}


const Histogram& JointDistributions::get(std::string attr1)
{
  std::vector<std::string> attributes(1);

  attributes[0] = attr1;

  return get(attributes);
}

const Histogram& JointDistributions::get(std::string attr1, std::string attr2)
{
  std::vector<std::string> attributes(2);

  attributes[0] = attr1;
  attributes[1] = attr2;

  return get(attributes);
}

const Histogram& JointDistributions::get(std::string attr1, std::string attr2, std::string attr3)
{
  std::vector<std::string> attributes(3);

  attributes[0] = attr1;
  attributes[1] = attr2;
  attributes[2] = attr3;

  return get(attributes);
}

bool JointDistributions::peek(const std::vector<std::string>& attributes){

  uint64_t code = 0;
  std::unordered_map<std::string,uint8_t>::iterator mIt;
  std::unordered_map<uint64_t,uint32_t>::iterator hIt;

  for (uint8_t i=0;i<attributes.size();i++) {
    // First we find the attribute index
    // std::cout<< "attr = "<<attributes[i]<<"\n";

    mIt = mNameMap.find(attributes[i]);
    if (mIt == mNameMap.end()) {
      fprintf(stderr,"Could not find attribute \"%s\"\n",attributes[i].c_str());

      return false;
    }
    code += (uint64_t)mIt->second << 4*i;//sizeof(uint8_t)*i;

  }
  // std::cout<<"code = "<<code<<"\n";

  hIt = mHistogramMap.find(code);
  if (hIt == mHistogramMap.end()) {
    fprintf(stderr,"Could not find histogram for for code %llu \n",code);

    return false;
    //assert (false);
  }
  return true;
  //return mHistograms[hIt->second];
}



const Histogram& JointDistributions::get(const std::vector<std::string>& attributes)
{
  uint64_t code = 0;
  std::unordered_map<std::string,uint8_t>::iterator mIt;
  std::unordered_map<uint64_t,uint32_t>::iterator hIt;

  for (uint8_t i=0;i<attributes.size();i++) {
    // First we find the attribute index
    // std::cout<< "attr = "<<attributes[i]<<"\n";

    mIt = mNameMap.find(attributes[i]);
    if (mIt == mNameMap.end()) {
      fprintf(stderr,"Could not find attribute \"%s\"\n",attributes[i].c_str());

      assert(false);
    }
    code += (uint64_t)mIt->second << 4*i;//sizeof(uint8_t)*i;

  }
  // std::cout<<"code = "<<code<<"\n";

  hIt = mHistogramMap.find(code);
  if (hIt == mHistogramMap.end()) {
    fprintf(stderr,"Could not find histogram for for code %llu \n",code);

    assert (false);
  }

  return mHistograms[hIt->second];
}

const Histogram& JointDistributions::get(const std::vector<uint8_t>& indices)
{
  uint64_t code = 0;
  std::unordered_map<uint64_t,uint32_t>::iterator hIt;

   for (uint8_t i=0;i<indices.size();i++)
     code += (uint64_t)indices[i] << 4*i;//sizeof(uint8_t)*i;

   // std::cout<<"code = "<<code<<"\n";

   hIt = mHistogramMap.find(code);
   if (hIt == mHistogramMap.end()) {
     fprintf(stderr,"Could not find histogram for for code %llu \n",code);

     assert (false);
   }

   return mHistograms[hIt->second];
}

void JointDistributions::computeHistograms(const HDData& data)
{
  std::vector<Histogram>::iterator it;

  for (uint64_t i=0;i<data.size();i++) {
    for (it=mHistograms.begin();it!=mHistograms.end();it++)
      it->addValue(data[i]);
  }

}

void JointDistributions::addValue(const float* sample)
{
  static std::vector<Histogram>::iterator it;
  for (it=mHistograms.begin();it!=mHistograms.end();it++)
    it->addValue(sample);
}


//////////// IO ////////////
void JointDistributions::serialize(std::ostream &output)
{ // file.open(filename, std::ios::in | std::ios::binary);
  cereal::BinaryOutputArchive binAR(output);
  // cereal::XMLOutputArchive binAR(output);
  binAR( this->mAttributes, this->mNameMap, this->mHistogramMap);

}

void JointDistributions::deserialize(std::istream &input){
  cereal::BinaryInputArchive binAR(input);
  // cereal::XMLInputArchive binAR(input);
  binAR( this->mAttributes, this->mNameMap, this->mHistogramMap);
}

bool JointDistributions::load(HDFileFormat::DistributionHandle &handle){
  std::vector<HDFileFormat::HistogramHandle> histoHandles;
  handle.getAllChildrenByType<HDFileFormat::HistogramHandle>(histoHandles);

  //load class states
  HDFileFormat::DataBlockHandle *bufferHandle = handle.getChildByType<HDFileFormat::DataBlockHandle>("serializedMetaData");
  //hderror(bufferHandle==NULL,"No serializedMetaData found");
  //de-serialize
  mSerializationBuffer.clear();
  mSerializationBuffer.resize(bufferHandle->sampleCount());
  bufferHandle->readData(&mSerializationBuffer[0]);

  autoResizeMembuf inBuffer(false);
  // membuf inBuffer(&mSerializationBuffer[0], mSerializationBuffer.size(), false);
  std::istream is(&inBuffer);
  inBuffer.setBuffer(mSerializationBuffer);
  this->deserialize(is);

  //load histogram
  mHistograms.resize(histoHandles.size());
  for(size_t i=0; i<histoHandles.size(); i++){
    mHistograms[i].load(histoHandles[i]);
  }
  return true;
}

bool JointDistributions::save(HDFileFormat::DistributionHandle &handle){

  //save class states
  /////// need serialization ////////
  // std::vector<std::string> mAttributes;
  // std::unordered_map<std::string,uint8_t> mNameMap;
  // std::unordered_map<uint64_t,uint32_t> mHistogramMap;

  // membuf streamBuffer(&mSerializationBuffer[0], mSerializationBuffer.size());
  // membuf streamBuffer(&mSerializationBuffer);
  autoResizeMembuf streamBuffer;
  // {
  std::ostream os(&streamBuffer);
  this->serialize(os);
  // }
  streamBuffer.copyBuffer(mSerializationBuffer);
  // fprintf(stderr, " == done serialization == %ld\n", streamBuffer.outputCount());

  HDFileFormat::DataBlockHandle bufferHandle;
  bufferHandle.idString("serializedMetaData");
  bufferHandle.setData(&mSerializationBuffer[0], streamBuffer.outputCount());
  handle.add(bufferHandle);

  //save histogram
  // /*
  for(size_t i=0; i<mHistograms.size(); i++){
      HDFileFormat::HistogramHandle histHandle;
      mHistograms[i].save(histHandle);
      handle.add(histHandle);
  }
  //*/

  return true;
}
