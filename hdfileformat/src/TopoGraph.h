/*
 * TopoGraph.h
 *
 *  Created on: Mar 8, 2015
 *      Author: bremer5
 */

#ifndef TOPOGRAPH_H
#define TOPOGRAPH_H

#include <vector>
#include <map>

#include "Definitions.h"
#include <stddef.h>

namespace HDFileFormat {

enum NodeType {
  MAXIMUM = 0,
  MINIMUM = 1,
  SADDLE = 2,
};

class TopoNode
{
public:

  class const_iterator
  {
  public:

    const_iterator() : mNode(NULL) {}

    const_iterator(const TopoNode* node, const std::vector<uint32_t>::const_iterator& it) : mNode(node),mIt(it) {}

    const_iterator(const const_iterator& it) : mNode(it.mNode), mIt(it.mIt) {}

    const_iterator& operator++(int i) {mIt++;return *this;}

    bool operator==(const const_iterator& it) const {return ((mNode->id() == it.mNode->id()) && (mIt == it.mIt));}

    bool operator!=(const const_iterator& it) const {return ((mNode->id() != it.mNode->id()) || (mIt != it.mIt));}

    const TopoNode& operator*() const {return mNode->node(*mIt);}

    const TopoNode* operator->() const {return &mNode->node(*mIt);}

  private:

    //! The reference to the node
    const TopoNode* mNode;

    //! The internal vector iterator
    std::vector<uint32_t>::const_iterator mIt;

  };


  //! Default constructor
  TopoNode(uint32_t id, float f, NodeType t);

  //! Copy constructor
  TopoNode(const TopoNode& node);

  //! Destructor
  ~TopoNode() {}

  //! Assignment operator
  TopoNode& operator=(const TopoNode& node);

  //! Return the begin of the neighobrs
  const_iterator begin() const {return const_iterator(this,mNeighbors.begin());}

  //! Return the begin of the neighobrs
  const_iterator end() const {return const_iterator(this,mNeighbors.end());}

  //! Return the global id
  uint32_t id() const {return mId;}

  //! Return the local index
  uint32_t index() const {return mIndex;}

  //! Return the node type
  HDFileFormat::NodeType type() const {return mType;}

  //! Return the function value
  float function() const {return mFunction;}

  //! Return the set of neighbors
  void neighbors(std::vector<const TopoNode* >& neighbors) const;

  //! Set the id
  void id(uint32_t id) {mId = id;}

  //! Set the index
  void index(uint32_t index) {mIndex = index;}

  //! Set the type
  void type(HDFileFormat::NodeType t) {mType = t;}

  //! Set the function value
  void function(float f) {mFunction = f;}

  //! Add a neighbor
  void addNeighbor(uint32_t node) {mNeighbors.push_back(node);}

private:

  //! The global index of this node
  uint32_t mId;

  //! The local index of this node
  uint32_t mIndex;

  //! The type of this node
  NodeType mType;

  //! The function value of this node
  float mFunction;

  //! A list of neighboring nodes
  std::vector<uint32_t> mNeighbors;

  const TopoNode& node(uint32_t index) const;
};


//! A generic topological graph
class TopoGraph
{
public:

  //! Typedef for the index map
  typedef std::map<uint32_t,uint32_t> IndexMap;

  //! Typedef node iterator
  typedef std::vector<TopoNode>::const_iterator const_iterator;

  //! Constructor
  TopoGraph() {}

  //! Destructor
  ~TopoGraph() {}

  //! Add a node and return it
  uint32_t addNode(uint32_t id, float f, NodeType t);

  //! Add an edge between two nodes
  void addEdge(uint32_t u, uint32_t v);

  //! Provide access to all nodes
  const std::vector<TopoNode>& nodes() const {return mNodes;}

  //! Return the number of node
  uint32_t size() const {return mNodes.size();}

  //! Return the begin of the nodes
  const_iterator begin() const {return mNodes.begin();}

  //! Return the end of the nodes
  const_iterator end() const {return mNodes.end();}

  //! Generate dot file
  void writeDot(const char* filename) const;

private:

  //! The list of nodes
  std::vector<TopoNode> mNodes;

  //! The map from global to local indices
  IndexMap mIndexMap;
};


}


#endif /* TOPOGRAPH_H_ */
