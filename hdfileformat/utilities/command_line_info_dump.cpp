/*
 * unit_test.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: Shusen Liu
 *
 *  Testing the read of volumeSeg
 *  Testing segmentation handle
 */

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <fstream>

#include "DataCollectionHandle.h"
#include "DatasetHandle.h"
#include "MorseComplex.h"
#include "SegmentationHandle.h"

using namespace HDFileFormat;

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
  DatasetHandle dataset = group.dataset(0);

  //check the segment associated with function
  FunctionHandle func = dataset.getFunction(0);
  SegmentationHandle seg = *func.getChildByType<SegmentationHandle>(0);
  //"Outlying", "Skewed", "Clumpy", "Sparse", "Striated", "Convex", "Skinny", "Stringy", "Monotonic"
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Outlying"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Skewed"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Stringy"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Monotonic"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-PPI_Hole"));

  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Stress"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-ClassSeparation"));
  
  //dump topo-info
  
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-"));

  MorseComplex morse;

  morse.initialize(seg);

  for (uint32_t i=0;i<morse.segCount();i++) {
    fprintf(stderr,"Extremum %5d, %0.4f   parent %5d   with persistence %0.4f\n",
    morse.mHierarchy[i].id,
    morse.mNodes[i].function,
    morse.mHierarchy[morse.mHierarchy[i].parent].id,
    morse.mHierarchy[i].parameter);
  }

  TopoGraph graph;

  //float persistence = 0.0005;
  float persistence = 0.005;
  //float ridgeness = 0.4;
  morse.cancellationTree(graph, persistence);
  //morse.topologicalSpine(graph, persistence, ridgeness);
  //morse.writeDot("test.dot");

  Segmentation segmentation;
  morse.segmentation(segmentation, persistence);

  printf("Graph size %ld\n", graph.nodes().size());

  TopoGraph::const_iterator it;
  TopoNode::const_iterator nIt;
  int totalPointSize=0;
  for (it=graph.begin();it!=graph.end();it++) {
    fprintf(stderr,"Node %d F(%f) T(%d): [",it->id(), it->function(),it->type());
    //fprintf(stderr,"Node %d F(%f): [",it->index(), it->function());

    //get segment
    if(it->type() != SADDLE)
    {
      Segment segment = segmentation.elementSegmentation(it->id());
      totalPointSize += segment.size;
      //fprintf(stderr,"<segment size: %d>",segment.size);
    }

    for (nIt=it->begin();nIt!=it->end();nIt++) {
      fprintf(stderr,"%d, ",nIt->id());
      //fprintf(stderr,"%d, F(%f)[d%f]",nIt->index(), nIt->function(), std::abs(f2-f1) );
    }
    fprintf(stderr,"]\n");
  }

  printf("Morse mOrder size: %ld\n", morse.mOrder.size());

  const std::vector<uint32_t> *offsets = segmentation.offsets();
  const std::vector<uint32_t> *samples = segmentation.segmentation();

  //dump class label
  std::vector<int> segLabel;
  segLabel.resize(totalPointSize);
  std::string inputFileName = std::string(argv[1]);
  std::string outputFileName = inputFileName.substr(0, inputFileName.find_last_of("."))+std::string("_label.csv");
  std::ofstream outputLabel(outputFileName.c_str());
  fprintf(stderr,"At persistence %f\n",persistence);
  int classLabel=0;
  for (uint32_t i=0;i<segmentation.segCount();i++)
  {
    int segmentSize = (*offsets)[i+1]-(*offsets)[i];
    fprintf(stderr,"\tSegment %d: size = %d\n",i,segmentSize);
    //dump class label
    for(int j=0; j<segmentSize; j++)
    {
      segLabel[(*samples)[ (*offsets)[i]+j ]] = classLabel;
      fprintf(stderr,"\t\t label:%d - index:%d\n", classLabel, (*samples)[ (*offsets)[i]+j ]);
    }
    classLabel++;
  }

  for(size_t i=0; i<segLabel.size(); i++)
    outputLabel << segLabel[i] << std::endl;

  outputLabel.close();

  return 1;
}


