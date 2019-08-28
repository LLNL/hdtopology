/*
 * MorseComplex.cpp
 *
 *  Created on: Feb 13, 2015
 *      Author: bremer5
 */

#include <algorithm>
#include <cmath>
#include "MorseComplex.h"

namespace HDFileFormat {

MorseComplex::MorseComplex()
{
}


int MorseComplex::initialize(SegmentationHandle& handle)
{
  HierarchicalSegmentation::initialize(handle);

  //access hierarchy
  DataBlockHandle* saddles = handle.getChildByType<DataBlockHandle>("SaddlePairs");

  hderror(saddles==NULL,"No hierarchy found");

  mSaddles.resize(saddles->sampleCount());

  saddles->readData(&mSaddles[0]);

  //access extrema graph
  DataBlockHandle* info = handle.getChildByType<DataBlockHandle>("NodeInfo");
  mNodes.resize(info->sampleCount());
  info->readData(&mNodes[0]);

  hderror(info==NULL,"No node info found");

  //make calculation for the Persistence plot
  this->mUpdatePersistencePlotData();

  return 1;
}

SegmentationHandle MorseComplex::makeHandle()
{
  SegmentationHandle handle;

  handle = HierarchicalSegmentation::makeHandle();

  DataBlockHandle block;

  block.idString("SaddlePairs");
  block.setData<SaddlePair>(&mSaddles[0],mSaddles.size());

  handle.add(block);

  DataBlockHandle block2;

  block2.idString("NodeInfo");
  block2.setData<NodeInfo>(&mNodes[0],mNodes.size());

  handle.add(block2);

  return handle;
}

void MorseComplex::addSaddlePair(uint32_t saddle, float f, uint32_t a, uint32_t b, float p)
{
  SaddlePair s;

  //fprintf(stderr,"SaddlePair %d   %d %d \n",saddle,a,b);
  s.saddle = saddle;
  s.function = f;
  s.neigh[0] = a;
  s.neigh[1] = b;
  s.parameter = p;

  mSaddles.push_back(s);
}

void MorseComplex::finalizeConstruction()
{
  std::sort(mSaddles.begin(),mSaddles.end(),pairCmp);
}

void MorseComplex::setNodeInfo(uint32_t global_index, float function)//, NodeType type)
{
  //fprintf(stderr,"MorseComplex::setNodeInfo global_index %d\n",global_index);
  uint32_t l = this->local(global_index);

  if (l >= mNodes.size())
    mNodes.resize(this->mHierarchy.size());

  mNodes[l].function = function;
  mNodes[l].type = MAXIMUM;
}


void MorseComplex::cancellationTree(TopoGraph& tree,float persistence) const
{
  uint32_t s = 0;
  uint32_t left,right;
  uint32_t saddle;
  uint32_t extremum;

  // Ignore all saddles not part of the tree (i.e. those that never
  // get cancelled)
  while ((s < mSaddles.size()) && (mSaddles[s].parameter > 10e33))
    s++;

  // For all the saddles that could be canceled but do exist at this
  // persistence threshold
  while ((s < mSaddles.size()) && (mSaddles[s].parameter > persistence)) {
    left = local(mSaddles[s].neigh[0]);
    left = representative(left,persistence);

    right = local(mSaddles[s].neigh[1]);
    right = representative(right,persistence);

    if (left != right) {
      saddle = tree.addNode(mSaddles[s].saddle,mSaddles[s].function,SADDLE);

      extremum = tree.addNode(this->mHierarchy[left].id,mNodes[left].function,mNodes[left].type);

      tree.addEdge(saddle,extremum);

      extremum = tree.addNode(this->mHierarchy[right].id,mNodes[right].function,mNodes[right].type);

      tree.addEdge(saddle,extremum);
    }

    s++;
  }
}


void MorseComplex::topologicalSpine(HDFileFormat::TopoGraph& tree,float persistence, float ridgeness) const
{
  uint32_t s =0;
  uint32_t left,right;
  uint32_t saddle;
  uint32_t extremum;
  std::vector<SaddlePair>::const_iterator it;
  float p;

  if (persistence > ridgeness) {
    ridgeness = persistence;
    fprintf(stderr,"Warning: ridgeness should be bigger than persistence.\n");
  }

  // We first construct the cancellation tree
  cancellationTree(tree,persistence);

  // Now examine all saddles not part of the cancellation tree
  while ((s < mSaddles.size()) && (mSaddles[s].parameter > 10e33)) {

    // Find the current left representative
    left = local(mSaddles[s].neigh[0]);
    left = representative(left,persistence);

    // Find the current right representative
    right = local(mSaddles[s].neigh[1]);
    right = representative(right,persistence);

    // If this is a strangulation suppress it
    if (left == right) {
      s++;
      continue;
    }

    // Compute the ridgeness of the saddle
    p = std::min(fabs(mNodes[left].function - mSaddles[s].function),
                 fabs(mNodes[right].function - mSaddles[s].function));

    // If it looks like a ridge
    if (p < ridgeness) {
      saddle = tree.addNode(mSaddles[s].saddle,mSaddles[s].function,SADDLE);
      extremum = tree.addNode(this->mHierarchy[left].id,mNodes[left].function,mNodes[left].type);
      tree.addEdge(saddle,extremum);

      extremum = tree.addNode(this->mHierarchy[right].id,mNodes[right].function,mNodes[right].type);
      tree.addEdge(saddle,extremum);

    }
    s++;
  }

}

float MorseComplex::getPersistenceByNodeNumber(int nodeNum)
{
  float persistence = 0.0;

  if(nodeNum>=mOrder.size()-1 || nodeNum <= 0)
  {
    nodeNum = mOrder.size()-2;
  }

  while(persistence<=0.0 && nodeNum > 1)
  {
    persistence = mHierarchy[ mOrder[mOrder.size()-nodeNum] ].parameter;
    nodeNum--;
  }

  assert(persistence>=0.0);
  assert(persistence==persistence);
  return persistence;
}


bool MorseComplex::pairCmp(const SaddlePair& a, const SaddlePair& b)
{
  return (a.parameter > b.parameter);
}

struct sort_mergeList {
    bool operator()(const std::pair<uint32_t,float> &left, const std::pair<uint32_t,float> &right) {
        return left.second < right.second;
    }
};

void MorseComplex::mUpdatePersistencePlotData()
{
  //create sorted list if mOrder is empty
  mOrder.clear();
  if(!mOrder.size())
  {
    std::vector<std::pair<uint32_t, float> > listOfMerges;
    for(size_t i=0; i<this->segCount(); i++)
      listOfMerges.push_back(std::pair<uint32_t, float>(i, mHierarchy[i].parameter));

    std::sort(listOfMerges.begin(), listOfMerges.end(), sort_mergeList());

    for(size_t i=0; i<listOfMerges.size(); i++)
    {
      mOrder.push_back(listOfMerges[i].first);
      //printf("order: %d ,%f\n",listOfMerges[i].first, listOfMerges[i].second);
    }

    //update the max persistence
    mMaxPersistence = listOfMerges[mOrder[mOrder.size()-2]].second;
    //printf("@@@@@@@@@@@@@@ Max Persistence @@@@@@@@@@@@ %f\n", mMaxPersistence);
  }

  //generate persistence vs. extrema retain in the graph
  mPersistenceVersusExtrema.clear();
  for(size_t i = 0; i<mOrder.size()-1; i++)
  {
    float persistence = mHierarchy[mOrder[i]].parameter;
    int numberOfExtrema = 0;
    for(size_t j=0; j<this->segCount(); j++)
      if(mHierarchy[j].parameter>persistence)
        numberOfExtrema++;
    //printf("persistence: %f, extrema: %d\n", persistence, numberOfExtrema);

    if(numberOfExtrema<=10)
    {
      mPersistenceVersusExtrema.push_back(std::pair<float, int>(persistence, numberOfExtrema) );
    }

  }

  //printf("\n");

  //generate persistence vs. number of arcs left
  mPersistenceVersusArcs.clear();
  for(size_t i = 0; i<mOrder.size()-1; i++)
  {
    float persistence = mHierarchy[mOrder[i]].parameter;
    int numberOfArcs = 0;
    for(size_t j=0; j<this->segCount(); j++)
      if(mHierarchy[j].parameter<persistence)
        numberOfArcs++;
    //printf("persistence: %f, arcs: %d\n", persistence, numberOfArcs);
    mPersistenceVersusArcs.push_back(std::pair<float, int>(persistence, numberOfArcs) );
  }

  //printf("\n\n\n =============== MorseComplex::Update persistence plot ============= \n Extrema %ld, Arcs %ld\n\n\n", mPersistenceVersusExtrema.size(), mPersistenceVersusArcs.size());
}

void MorseComplex::writeDot(const char* filename) const
{
  FILE* output = fopen(filename,"w");


  fprintf(output,"graph G {\n");
  fprintf(output,"\tnode [shape=plaintext,fontsize=10];\n");
  fprintf(output,"\tedge [color=black,len=2];\n");

  for (uint32_t i=0;i<this->mNodes.size();i++) {
    fprintf(output,"%d [label=\"%d,  %0.3f\",shape=circle]\n",this->mHierarchy[i].id,this->mHierarchy[i].id,mNodes[i].function);
  }

  for (uint32_t i=0;i<this->mSaddles.size();i++) {
    fprintf(output,"%d [label=\"%d, %0.3f\",shape=box]\n",mSaddles[i].saddle,mSaddles[i].saddle,mSaddles[i].function);
  }


  for (uint32_t i=0;i<mSaddles.size();i++) {
    fprintf(output,"%d -- %d\n",mSaddles[i].saddle,mSaddles[i].neigh[0]);
    fprintf(output,"%d -- %d\n",mSaddles[i].saddle,mSaddles[i].neigh[1]);
  }

  fprintf(output,"}\n");
  fclose(output);
}


std::istream& operator>>(std::istream &input, MorseComplex::SaddlePair &s)
{
  input >> s.saddle;
  input >> s.function;
  input >> s.neigh[0];
  input >> s.neigh[1];
  input >> s.parameter;

  return input;
}

std::ostream& operator<<(std::ostream &output, MorseComplex::SaddlePair &s)
{
  output << s.saddle << " " << s.function << " " << s.neigh[0] << " " << s.neigh[1] << " " << s.parameter;

  return output;
}


std::istream& operator>>(std::istream &input, MorseComplex::NodeInfo &info)
{
  input >> info.function;

  uint8_t t;
  input >> t;
  info.type = (NodeType)t;

  return input;
}

std::ostream& operator<<(std::ostream &output, MorseComplex::NodeInfo &info)
{
  output << info.function << " " << info.type;

  return output;
}


}
