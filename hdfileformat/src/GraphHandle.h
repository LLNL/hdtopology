#ifndef NEIGHBOR_HOOD_GRAPH_HANDLE_H
#define NEIGHBOR_HOOD_GRAPH_HANDLE_H

#include "DataBlockHandle.h"

namespace HDFileFormat{

/*
 * Storage a graph, edge connectivity + edge weight
 * Author: Shusen Liu Date: Oct 14, 2014
*/  

enum GraphStorageType
{
  GRAPH_STORAGE_TYPE_EDGE_PAIR,
  GRAPH_STORAGE_TYPE_DAG,
  GRAPH_STORAGE_UNDEFINED
};
  
class GraphHandle : public DataBlockHandle
{
public:

  friend class DataCollectionHandle;

  //! The default name of the data set
  static const std::string sDefaultGraphName;

  //! Constructor
  explicit GraphHandle(HandleType t=H_GRAPH);

  //! Constructor
  GraphHandle(const char *filename, HandleType t=H_GRAPH);

  //! Copy constructor
  GraphHandle(const GraphHandle& handle);

  //! Destructor
  virtual ~GraphHandle();

  //! Assignment operator
  GraphHandle & operator=(const GraphHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new GraphHandle(*this);}
  
  //! add children
  virtual FileHandle& add(const FileHandle &handle);
  
  //! convenience function for adding edges
  void setEdgePairs(uint32_t *edgePair, uint32_t edgePairCount);

  //! adding edge weight
  void setEdgeWeight(float *edgeWeight, int edgeCount);

  //! convenience function for geting edge count
  int edgePairNum(){return mSampleCount;}

protected:
  GraphStorageType mGraphType;

  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;
  
};


}

#endif
