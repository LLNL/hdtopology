/*
 * unit_test.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: bremer5
 */

#include <vector>
#include <string>

#include "DataCollectionHandle.h"
#include "DatasetHandle.h"
#include "DataBlockHandle.h"
#include "ClusterHandle.h"
#include "EmbeddingHandle.h"
#include "DataPointsHandle.h"
#include "GraphHandle.h"

/*! Testing write dataset
 * 
*/


using namespace HDFileFormat;

int main(int argc, const char* argv[])
{
  DataPointsHandle dataPoints;
  DatasetHandle dataset;
  ClusterHandle cluster;
  
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
  
  //add clustering results
  //write the cluster label
  uint32_t* label = new uint32_t[pointCount];
  for(int i=0; i<pointCount; i++)
    label[i] = i;
  cluster.setLabel(label, pointCount);

  DataCollectionHandle group("test.hdff");
  
  HierarchyHandle hierarchy;
  SubspaceHandle subspace;
    
  //add basis
  hierarchy.setHierarchy(label, pointCount);
  std::vector<Basis> basisVec;
  Basis b1(2,2), b2(1,2);
  b1.Coeff(0,0) = 1.0; b1.Coeff(0,1) = 1.0;
  b2.Coeff(0,0) = 1.0;  
  basisVec.push_back(b1);
  basisVec.push_back(b2);
  
  subspace.setChildrenBasis(basisVec);
  cluster.add(hierarchy).add(subspace);

  //add embedding
  EmbeddingHandle embedding;
  BasisHandle basis;
  basis.setBasis(b1);
  embedding.add(basis);
  embedding.setData(label, pointCount, 1);
  
  //add neighborhood graph
  GraphHandle graph;
  graph.setEdgePairs(label, pointCount/2);  
  
  //create the dataset
  dataset.add(dataPoints);
  dataset.add(embedding);
  dataset.add(cluster);
  dataset.add(graph);
  //dataset.add(dataPoints);
  
  group.add(dataset);
  group.write();
  
  delete[] data;
  delete[] label;
}

