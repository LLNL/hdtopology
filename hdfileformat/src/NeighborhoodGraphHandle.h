#ifndef NEIGHBOR_HOOD_GRAPH_HANDLE_H
#define NEIGHBOR_HOOD_GRAPH_HANDLE_H

#include "FileHandle.h"

namespace HDFileFormat{

class NeighborhoodGraphHandle : public FileHandle
{
public:

  friend class DataCollectionHandle;

  //! The default name of the data set
  static const std::string sDefaultPointSetName;

  //! Constructor
  NeighborhoodGraphHandle(HandleType t=H_DATAPOINTS);

  //! Constructor
  NeighborhoodGraphHandle(const char *filename, HandleType t=H_DATAPOINTS);

  //! Copy constructor
  NeighborhoodGraphHandle(const NeighborhoodGraphHandle& handle);

  //! Destructor
  virtual ~NeighborhoodGraphHandle();

  //! Assignment operator
  NeighborhoodGraphHandle & operator=(const NeighborhoodGraphHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new NeighborhoodGraphHandle(*this);}

protected:

  //! A void pointer to the data used for writing
  uint32_t* mGraphEdgePair;

  //! Name of the dataset
  std::string mNeighborGraphName;

  //! Reset all values to their default uninitialized values
  void clear();

  //! Parse the xml tree
  virtual int parseXML(const XMLNode& node) {return parseXMLInternal(node);}

  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;
  
  
};


}

#endif
