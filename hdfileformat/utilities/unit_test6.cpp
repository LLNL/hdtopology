/*
 * unit_test.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: Shusen Liu
 *
 *  Testing the read of volumeSeg
 *
 */

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

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

  //FunctionHandle func = dataset.getFunction(0);
  //SegmentationHandle seg = *func.getChildByType<SegmentationHandle>(0);
  //"Outlying", "Skewed", "Clumpy", "Sparse", "Striated", "Convex", "Skinny", "Stringy", "Monotonic"
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Outlying"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Skewed"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Stringy"));
  SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Monotonic"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-PPI_Hole"));

  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-Stress"));
  //SegmentationHandle seg =  *dataset.getChildInFullHierarchyByType<SegmentationHandle>(std::string("MC-ClassSeparation"));

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
  float persistence = 0.1;
  float ridgeness = 0.2;
  //morse.cancellationTree(graph, persistence);
  morse.topologicalSpine(graph, persistence, ridgeness);
  //morse.writeDot("test.dot");

  Segmentation segmentation;
  morse.segmentation(segmentation, persistence);

  printf("Graph size %ld\n", graph.nodes().size());

  TopoGraph::const_iterator it;
  TopoNode::const_iterator nIt;
  for (it=graph.begin();it!=graph.end();it++) {
    fprintf(stderr,"Node %d F(%f) T(%d): [",it->id(), it->function(),it->type());
    //fprintf(stderr,"Node %d F(%f): [",it->index(), it->function());

    //get segment
    if(it->type() != SADDLE)
    {
      Segment segment = segmentation.elementSegmentation(it->id());
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

  fprintf(stderr,"At persistence %f\n",persistence);
  for (uint32_t i=0;i<segmentation.segCount();i++) {
    fprintf(stderr,"\tSegment %d: size = %d\n",i,(*offsets)[i+1]-(*offsets)[i]);
  }


  return 1;
}


