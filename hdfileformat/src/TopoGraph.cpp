/*
 * TopoNode.cpp
 *
 *  Created on: Mar 8, 2015
 *      Author: bremer5
 */

#include "TopoGraph.h"
#include "stdio.h"

namespace HDFileFormat {

TopoNode::TopoNode(uint32_t id, float f, NodeType t) : mId(id), mIndex(-1),
    mType(t), mFunction(f)
{
}

TopoNode::TopoNode(const TopoNode& node)
{
  mId = node.mId;
  mIndex = node.mIndex;
  mType = node.mType;
  mFunction = node.mFunction;
  mNeighbors = node.mNeighbors;
}

TopoNode& TopoNode::operator=(const TopoNode& node)
{
  mId = node.mId;
  mIndex = node.mIndex;
  mType = node.mType;
  mFunction = node.mFunction;
  mNeighbors = node.mNeighbors;

  return *this;
}

void TopoNode::neighbors(std::vector<const TopoNode*>& neighbors) const
{
  neighbors.resize(mNeighbors.size());

  for (uint32_t i=0;i<mNeighbors.size();i++)
    neighbors[i] = &node(mNeighbors[i]);
}

const TopoNode& TopoNode::node(uint32_t index) const
{
  return *(this + index - mIndex);
}

uint32_t TopoGraph::addNode(uint32_t id, float f, NodeType t)
{
  IndexMap::iterator it;

  //fprintf(stderr,"TopoGraph::addNode %d %d\n",id,t);

  it = mIndexMap.find(id);

  if ((it != mIndexMap.end()) && (t != SADDLE)) {

    hderror(mNodes[it->second].id() != id,"Nodes inconsistent");
    hderror(mNodes[it->second].function() != f,"Nodes inconsistent");
    hderror(mNodes[it->second].type() != t,"Nodes inconsistent");

    return it->second;
  }
  else {
    mNodes.push_back(TopoNode(id,f,t));
    mNodes.back().index(mNodes.size()-1);
    mIndexMap[id] = mNodes.size()-1;

    return mNodes.size()-1;
  }
}

void TopoGraph::addEdge(uint32_t u, uint32_t v)
{
  //fprintf(stderr,"TopoGraph::addEdge %d -> %d\n",mNodes[u].id(),mNodes[v].id());

  mNodes[u].addNeighbor(v);
  mNodes[v].addNeighbor(u);
}

void TopoGraph::writeDot(const char* filename) const
{
  FILE* output = fopen(filename,"w");
  TopoNode::const_iterator nIt;


  fprintf(output,"graph G {\n");
  fprintf(output,"\tnode [shape=plaintext,fontsize=10];\n");
  fprintf(output,"\tedge [color=black];\n");

  for (uint32_t i=0;i<mNodes.size();i++) {
    fprintf(output,"%d [label=\"%d\"]\n",i,mNodes[i].id());
    for (nIt=mNodes[i].begin();nIt!=mNodes[i].end();nIt++) {
      if (mNodes[i].id() < nIt->id()) {
        fprintf(output,"%d -- %d\n",i,nIt->index());
      }
    }
  }

  fprintf(output,"}\n");
  fclose(output);


}



}


