#include "HierarchyHandle.h"

namespace HDFileFormat{

const std::string HierarchyHandle::sDefaultHierarchyName = "Hierarchy";

//const std::string HierarchyHandle::sHierarchyTypeExplicit = "ExplicitHierarchy";

//const std::string HierarchyHandle::sHierarchyTypeImplicit = "ImplicitHierarchy";

HierarchyHandle::HierarchyHandle(HandleType t)
  :DataBlockHandle(t),
   mHierarchyType(HIERARCHY_EXPLICIT)
{
  mID = sDefaultHierarchyName;
}

HierarchyHandle::HierarchyHandle(const char *filename, HandleType t)
  :DataBlockHandle(filename, t),
   mHierarchyType(HIERARCHY_EXPLICIT)
{
  mID = sDefaultHierarchyName;
}

//always pass in the derived object to based class constructor
HierarchyHandle::HierarchyHandle(const HierarchyHandle &handle) 
  :DataBlockHandle(handle),
   mHierarchyType(handle.mHierarchyType)
{

}

HierarchyHandle::~HierarchyHandle()
{
  
}

HierarchyHandle &HierarchyHandle::operator=(const HierarchyHandle &handle)
{
  DataBlockHandle::operator=(handle);
  mHierarchyType = handle.mHierarchyType;
  //mHierarchyTypeString = handle.mHierarchyTypeString;
  
  return *this;
}
  
void HierarchyHandle::setHierarchy(uint32_t *hierarchyIndex, uint32_t indexLength, HierarchyStorageType type)
{
  mHierarchyType = type;
  
  //set the buffer
  DataBlockHandle::setData(hierarchyIndex, indexLength, 1);
  
}

int HierarchyHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);  
  
  // in order to support >> in FileHandle class
  // >> enum is not valid
  uint32_t type;
  getAttribute(node, "hierarchyType", type);
  mHierarchyType = static_cast<HierarchyStorageType>(type);

  return 1;
}

int HierarchyHandle::attachXMLInternal(XMLNode &node) const
{
  DataBlockHandle::attachXMLInternal(node);
  
  uint32_t type = static_cast<uint32_t>(mHierarchyType);
  addAttribute(node, "hierarchyType", type);
  
  return 1;
}


}//namespace
