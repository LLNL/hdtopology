/*
 * unit_test_compression.cpp
 *
 *  Created on: Mar 5, 2015
 *      Author: Shusen Liu
 *
 *  Test compression of block data (for volume label, metaInfo or large table)
 *
 */

#include <vector>
#include <string>

#include "DataCollectionHandle.h"
#include "DatasetHandle.h"
#include "DataBlockHandle.h"
#include "DataPointsHandle.h"
#include "GraphHandle.h"

/*! Testing write dataset
 *
*/

using namespace HDFileFormat;

int main(int argc, const char* argv[])
{
//#ifdef WRITE
  DataPointsHandle dataPoints;
  DatasetHandle dataset;

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
  dataPoints.compressionFlag(false);

  DataCollectionHandle group("testCompression.hdff");

  //create the dataset
  dataset.add(dataPoints);

  group.add(dataset);
  group.write();

  delete[] data;

//#endif

  ///////////// read ////////////
  DataCollectionHandle collectionRead;
  collectionRead.attach("testCompression.hdff");
  DatasetHandle &rDataset = collectionRead.dataset(0);
  DataPointsHandle *rDataPoints;
  rDataPoints = rDataset.getChildByType<DataPointsHandle>(0);

  float* rdata = new float[rDataPoints->sampleCount()*rDataPoints->dimension()];

  rDataPoints->readData(rdata);

  fprintf(stderr,"Read compression\n");
  fprintf(stderr,"\tFirst sample: \"");
  for (uint32_t i=0;i<rDataPoints->dimension();i++)
    fprintf(stderr,"%f ",rdata[i]);
  fprintf(stderr,"\"\n\n");

  delete[] rdata;
}

