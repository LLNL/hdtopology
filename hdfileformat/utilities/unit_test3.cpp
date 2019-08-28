/*
 * unit_test3.cpp
 *
 *  Created on: Nov 1, 2014
 *      Author: Shusen Liu
 */

#include <vector>
#include <string>
#include <iostream>

#include "DataCollectionHandle.h"
#include "DatasetHandle.h"
#include "DataBlockHandle.h"
#include "DataPointsHandle.h"
#include "ClusterHandle.h"
#include "GraphHandle.h"

using namespace HDFileFormat;

/*! Testing read dataset and add entry and write again
 *  Test "Save as" property
*/

int main(int argc, const char* argv[])
{
  const char* testFileName = "testEdit.hdff";
  const char* afterEditFileName = "testAfterEdit.hdff";
  ///////////////////////////// read ///////////////////////////
  
  DataPointsHandle dataPointsWrite;
  DatasetHandle datasetWrite;
  
  int pointCount = 50;
  int dimension = 10;
  //add the dataPoints
  float* dataWrite = new float[dimension*pointCount];
  for (int i=0;i<dimension*pointCount;i++)
    dataWrite[i] = float(i);

  dataPointsWrite.setData(dataWrite, pointCount, dimension);
  std::vector<std::string> emptyLabel;
  std::vector<bool> emptyFlag;
  dataPointsWrite.setDimensionLabel(emptyLabel);
  dataPointsWrite.setDimensionFlag(emptyFlag);
  dataPointsWrite.setSpatialDim(2,25,1);
  
  DataCollectionHandle group(testFileName);
  
  datasetWrite.add(dataPointsWrite);
  group.add(datasetWrite);
  group.write();
  
  delete[] dataWrite;
  ///////////////////////////// read ///////////////////////////
  
  DataCollectionHandle collection;
  if(collection.attach(testFileName) == 0) 
  {
    fprintf(stderr,"Could not attach to file \"%s\"\n",argv[1]);
    exit(0);
  }
  
  //assignment is not working correctly
  DatasetHandle &dataset = collection.dataset(0);
  int numberOfDataset = collection.datasetCount();
  printf("Dataset Count: %d\n", numberOfDataset);
  
  DataPointsHandle dataPoints;
  
  //test get all children function
  std::vector<DataPointsHandle> dataPointsVec;
  dataset.getAllChildrenByType(dataPointsVec);
  printf("dataPointsVec: size[%ld]\n", dataPointsVec.size());
  
  //output points
  dataset.getFirstChildByType(dataPoints);
  assert(dataPoints.isValid());
  fprintf(stderr,"Got dataset with\n");
  fprintf(stderr,"\t%d samples\n",dataPoints.sampleCount());
  fprintf(stderr,"\tof dimension %d\n",dataPoints.dimension());
  fprintf(stderr,"\tof type %s\n",dataPoints.dataType().c_str());
  fprintf(stderr,"\tof size %d bytes\n",dataPoints.valueSize());

  uint32_t dimX, dimY, dimZ;
  dataPoints.getSpatialDim(dimX,dimY,dimZ);
  printf("\tVolume Spatial Dim(%d): %d, %d, %d\n", dataPoints.spatialDim(), dimX, dimY, dimZ);
  std::vector<bool> dimensionFlag = dataPoints.dimensionFlags();
  std::cout << "\tDimensionFlag: ";
  for (int i=0; i<dimensionFlag.size();i++){  std::cout << dimensionFlag[i] << " ";  }  
  std::cout << std::endl;

  float* data = new float[dataPoints.sampleCount()*dataPoints.dimension()];

  dataPoints.readData(data);

  fprintf(stderr,"\tFirst sample: \"");
  for (uint32_t i=0;i<dataPoints.dimension();i++)
    fprintf(stderr,"%f ",data[i]);
  fprintf(stderr,"\"\n\n");

  //add cluster
  ClusterHandle cluster;
  uint32_t* label = new uint32_t[pointCount];
  for(int i=0; i<pointCount; i++)
    label[i] = i;
  cluster.setLabel(label, pointCount);
  
  dataset.add(cluster);
  
  //update the dataset
  //dimensionFlag[0] = false;
  //dataPoints.setDimensionFlag(dimensionFlag);
  
  //test save as
  collection.write(afterEditFileName);
  
  delete[] data;
  delete[] label;
  
  return 1;
}


