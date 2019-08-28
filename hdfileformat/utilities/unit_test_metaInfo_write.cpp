/*
 * unit_test_metaInfo_write.cpp
 *
 *  Created on: Mar 5, 2015
 *      Author: Shusen Liu
 *
 *  Testing the writing of MetaInfo
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
  const char* fileName = "testMetaInfo.hdff";

  DataPointsHandle dataPoints;
  DatasetHandle dataset;
  DataPointsMetaInfoHandle metaInfo;
  PointSetMetaInfo meta;

  int pointCount = 50;
  int dimension = 10;

  //add the dataPoints
  float* data = new float[dimension*pointCount];
  for (int i=0;i<dimension*pointCount;i++)
    data[i] = float(i);

  dataPoints.setData(data, pointCount, dimension);
  std::vector<std::string> emptyLabel;
  std::vector<bool> emptyFlag;
  dataPoints.setDimensionLabel(emptyLabel);
  dataPoints.setDimensionFlag(emptyFlag);
  dataPoints.setSpatialDim(2,25,1);
  dataPoints.idString(std::string("firstWrite"));

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

  meta.SetImageData(imageList, imageX, imageY);

  metaInfo.SetMetaInfo(meta);

  DataCollectionHandle group(fileName);
  dataPoints.add(metaInfo);
  dataset.add(dataPoints);

  //// test decompression ////
  group.add(dataset);
  group.write();

  delete[] data;

  return 0;
}


