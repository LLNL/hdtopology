#ifndef HIERARCHY_HANDLE_H
#define HIERARCHY_HANDLE_H

#include "DataBlockHandle.h"

/*
 * Hierarchy's leaf corresponding to the parent's cluster
 * Author: Shusen Liu Date: Oct 10, 2014
*/

namespace HDFileFormat{

enum HierarchyStorageType
{
  HIERARCHY_IMPLICIT,
  HIERARCHY_EXPLICIT
};

class HierarchyHandle: public DataBlockHandle
{
public:
  //! The default name of the data set
  static const std::string sDefaultHierarchyName;
  
  //! The hierarchy type strings
  //  static const std::string sHierarchyTypeExplicit;
  //  static const std::string sHierarchyTypeImplicit;
  
  //! Constructor
  explicit HierarchyHandle(HandleType t=H_HIERARCHY);

  //! Constructor
  HierarchyHandle(const char *filename, HandleType t=H_HIERARCHY);

  //! Copy constructor
  HierarchyHandle(const HierarchyHandle&);

  //! Destructor
  virtual ~HierarchyHandle();

  //! Assignment operator
  HierarchyHandle &operator=(const HierarchyHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new HierarchyHandle(*this);}
  
  //! Set raw buffer for different kind of hierarchy
  void setHierarchy(uint32_t* hierarchyIndex, uint32_t indexLength, HierarchyStorageType type =HIERARCHY_IMPLICIT);
  
  //! Get the index buffer size (for reading the raw data, do we really need this?)
  uint32_t indexLength(){return mSampleCount;}
  
  //! Require the hierarchy type
  HierarchyStorageType hierarchyType() {return mHierarchyType;}
  
  //! return the leaf number of the hierarchy
  int LeafNum();

protected:
  HierarchyStorageType mHierarchyType;
  
protected:
  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;
  
};

} //HDFileFormat namespace



#endif
