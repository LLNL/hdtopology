/*
 * FunctionHandle.cpp
 *
 *  Created on: Feb 15, 2015
 *      Author: bremer5
 */

#include <sstream>

#include "FunctionHandle.h"


namespace HDFileFormat {

FunctionHandle::FunctionHandle(const char* filename,HandleType t) : FileHandle(filename,t)
{
}

FunctionHandle::FunctionHandle(const FunctionHandle& handle) : FileHandle(handle)
{
  mDomain = handle.mDomain;
  mRange = handle.mRange;
}

FunctionHandle& FunctionHandle::operator=(const FunctionHandle& handle)
{
  FileHandle::operator=(handle);

  mDomain = handle.mDomain;
  mRange = handle.mRange;

  return *this;
}


FileHandle& FunctionHandle::add(const FileHandle& handle)
{
  if ((handle.type() != H_SEGMENTATION) && (handle.type() != H_GRAPH))
    hderror(true,"A FunctionHandle does not accept a \"\" handle as child",
            handle.typeName());

  mChildren.push_back(handle.clone());
  return *mChildren.back();
}


int FunctionHandle::attachXMLInternal(XMLNode& node) const
{
  FileHandle::attachXMLInternal(node);

  std::stringstream domain;

  domain << mDomain[0];

  for (uint32_t i=1;i<mDomain.size();i++)
    domain << " " << mDomain[i];

  addAttribute(node,"domain", domain.str());
  addAttribute(node,"range", mRange);

  return 1;
}


//! Parse the local information from the xml tree
int FunctionHandle::parseXMLInternal(const XMLNode& node)
{
  FileHandle::parseXMLInternal(node);

  getAttribute(node,"range",mRange);

  std::string domain_str;

  getAttribute(node,"domain",domain_str);
  std::stringstream domain(domain_str);

  uint32_t a;
  while (domain >> a)
    mDomain.push_back(a);




  return 1;
}


}


