/*
 * unit_test.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: bremer5
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

/*! Testing read dataset
 * 
*/

int main(int argc, const char* argv[])
{
  DataCollectionHandle group;  

  if(argv[1])
  {
    if(group.attach(argv[1]) == 0) {
      fprintf(stderr,"Could not attach to file \"%s\"\n",argv[1]);
      exit(0);
    }
  }
  else
  {
    fprintf(stderr, "No input file\n");
    exit(0);
  }
  
  //assignment is not working correctly
  DatasetHandle &dataset = group.dataset(0);
  int numberOfDataset = group.datasetCount();
  printf("Dataset Count: %d\n", numberOfDataset);
  
  DataPointsHandle dataPoints;
  ClusterHandle cluster;
  
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

  //output cluster
  dataset.getFirstChildByType(cluster);
  assert(cluster.isValid());
  fprintf(stderr,"\tCluster name: %s \n",cluster.idString().c_str());
  fprintf(stderr,"\t%d samples\n",cluster.sampleCount());
  
  int* label = new int[cluster.sampleCount()];
  cluster.readData(label);
  
  fprintf(stderr,"\tLabels: ");
  for(uint32_t i=0; i<cluster.sampleCount(); i++)
    fprintf(stderr,"%d ", label[i]);
  fprintf(stderr, "\n");
  
  //output subspace
  SubspaceHandle subspace;
  cluster.getFirstChildByType(subspace);
  assert(subspace.isValid());
  //fprintf(stderr, "\t\tSubspace name: %s\n",subspace.idString().c_str());
  fprintf(stderr, "\t\tSubspace number: %d\n",subspace.subspaceNum());
  for(int i=0; i<subspace.subspaceNum(); i++)
    std::cout<<"\t\t\tBasis:"<<subspace.subspaceBasisByIndex(i)<<std::endl;

  //output hierarchy
  HierarchyHandle hierarchy;
  cluster.getFirstChildByType(hierarchy);
  assert(hierarchy.isValid());
  //fprintf(stderr, "\t\tHierarchy name: %s\n",hierarchy.idString().c_str());
  fprintf(stderr, "\t\tHierarchy length: %d\n",hierarchy.indexLength());
  uint32_t* pHierarchy = new uint32_t[hierarchy.indexLength()];
  hierarchy.readData(pHierarchy);
  
  fprintf(stderr,"\t\tHierarchy: ");
  for(uint32_t i=0; i<hierarchy.indexLength(); i++)
    fprintf(stderr,"%d ", pHierarchy[i]);
  fprintf(stderr, "\n");
  
  //output graph
  GraphHandle graph;
  dataset.getFirstChildByType(graph);
  assert(graph.isValid());
  uint32_t* pEdges = new uint32_t[graph.edgePairNum()*2];
  graph.readData(pEdges);
  
  fprintf(stderr,"\tGraph: EdgesNum:%d  ", graph.edgePairNum());
  for(uint32_t i=0; i<graph.edgePairNum(); i++)
    fprintf(stderr,"[%d,%d] ", pEdges[2*i], pEdges[2*i+1]);
  fprintf(stderr, "\n");
  
  delete[] data;
  delete[] label;
  delete[] pHierarchy;
  delete[] pEdges;

  //update the dataset
  dimensionFlag[0] = false;
  dataPoints.setDimensionFlag(dimensionFlag);
  
  group.updateMetaData();
  
  return 1;
}


