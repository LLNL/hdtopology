/*
 * unit_test_metaInfo_read.cpp
 *
 *  Created on: Mar 5, 2015
 *      Author: Shusen Liu
 *
 *  Testing the read of metaInfo
 *
 */

#include <vector>
#include <string>
#include <iostream>

#include "DataCollectionHandle.h"
#include "DatasetHandle.h"
#include "DataPointsHandle.h"
#include "DataPointsMetaInfoHandle.h"
#include "PointSetMetaInfo.h"


using namespace HDFileFormat;

int main(int argc, const char* argv[])
{
  bool testFlag = true;

  DataCollectionHandle group;

  const char* fileName = "testMetaInfo.hdff";

  if(group.attach(fileName) == 0) {
    fprintf(stderr,"Could not attach file \"%s\"\n",fileName);
    exit(0);
  }

  //assignment is not working correctly
  DatasetHandle &dataset = group.dataset(0);
  int numberOfDataset = group.datasetCount();
  printf("Dataset Count: %d\n", numberOfDataset);

  DataPointsHandle *dataPoints = NULL;

  //test get all children function
  std::vector<DataPointsHandle> dataPointsVec;
  dataset.getAllChildrenByType(dataPointsVec);
  printf("dataPointsVec: size[%ld]\n", dataPointsVec.size());

  //output points
  dataPoints  = dataset.getChildByType<DataPointsHandle>(std::string("firstWrite"));
  assert(dataPoints->isValid());
  fprintf(stderr,"Got dataset with\n");
  fprintf(stderr,"\t%d samples\n",dataPoints->sampleCount());
  fprintf(stderr,"\tof dimension %d\n",dataPoints->dimension());
  fprintf(stderr,"\tof type %s\n",dataPoints->dataType().c_str());
  fprintf(stderr,"\tof size %d bytes\n",dataPoints->valueSize());

  uint32_t dimX, dimY, dimZ;
  dataPoints->getSpatialDim(dimX,dimY,dimZ);
  printf("\tVolume Spatial Dim(%d): %d, %d, %d\n", dataPoints->spatialDim(), dimX, dimY, dimZ);
  std::vector<bool> dimensionFlag = dataPoints->dimensionFlags();
  std::cout << "\tDimensionFlag: ";
  for (int i=0; i<dimensionFlag.size();i++){  std::cout << dimensionFlag[i] << " ";  }
  std::cout << std::endl;

  float* data = new float[dataPoints->sampleCount()*dataPoints->dimension()];

  dataPoints->readData(data);

  fprintf(stderr,"\tFirst sample: \"");
  for (uint32_t i=0;i<dataPoints->dimension();i++)
  {
    if(data[i] != i)
      testFlag = false;
    fprintf(stderr,"%f ",data[i]);
  }
  fprintf(stderr,"\"\n\n");

  ///////////// read metaInfo ////////////
  DataPointsMetaInfoHandle metaInfo;
  dataPoints->getFirstChildByType(metaInfo);
  assert(metaInfo.isValid());
  PointSetMetaInfo meta;
  metaInfo.ReadMetaInfo(meta);

  std::vector<std::vector<int> > imageList;
  meta.GetImageData(imageList);

  fprintf(stderr,"\tMeta Info: list len:%ld, dimX:%d, dimY:%d, channel:%d\n", imageList.size(), meta.GetImageX(), meta.GetImageY(), meta.GetChannelNum());

  std::vector<int> image = imageList[0];
  for(size_t i=0; i<image.size(); i++)
  {
    if(image[i] != i)
      testFlag = false;
    fprintf(stderr, "%d ", image[i]);
  }
  fprintf(stderr, "\n");

  if(meta.GetMetaInfoType() == META_INFO_IMAGE)
    fprintf(stderr, "\tImage\n");
  else if(meta.GetMetaInfoType() == META_INFO_STRING)
    fprintf(stderr, "\tString\n");

  delete[] data;

  if(testFlag)
    return 0;
  else
    return -1;
}


