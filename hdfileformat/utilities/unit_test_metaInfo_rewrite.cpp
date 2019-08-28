/*
 * unit_test_metaInfo_write.cpp
 *
 *  Created on: Mar 5, 2015
 *      Author: Shusen Liu
 *
 *  Testing the re-writing of MetaInfo
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

  //DataPointsHandle dataPoints;
  //dataset.getFirstChildByType(dataPoints);
  DataPointsHandle *dataPoints = dataset.getChildByType<DataPointsHandle>(std::string("firstWrite"));

  int pointCount = dataPoints->sampleCount();

  int imageX = 5;
  int imageY = 5;
  std::vector<std::vector<int> > imageList;
  for(int i=0; i<pointCount; i++)
  {
    std::vector<int> image;
    for(int j=0; j<imageX*imageY; j++)
    {
      image.push_back(j);
    }
    imageList.push_back(image);
  }

  PointSetMetaInfo meta;
  meta.SetImageData(imageList, imageX, imageY);

  DataPointsMetaInfoHandle metaInfo;
  metaInfo.SetMetaInfo(meta);

  dataPoints->append(metaInfo);

  //add the dataPoints
  int dimension = 10;
  DataPointsHandle newDataPoints;
  float* data = new float[dimension*pointCount];
  for (int i=0;i<dimension*pointCount;i++)
    data[i] = float(i);

  newDataPoints.setData(data, pointCount, dimension);
  std::vector<std::string> emptyLabel;
  std::vector<bool> emptyFlag;
  newDataPoints.setDimensionLabel(emptyLabel);
  newDataPoints.setDimensionFlag(emptyFlag);
  newDataPoints.setSpatialDim(2,25,1);
  newDataPoints.idString(std::string("appendedDataPoints"));
  dataset.append(newDataPoints);

  return 0;
}


