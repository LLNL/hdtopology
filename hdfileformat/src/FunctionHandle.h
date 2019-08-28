/*
 * FunctionHandle.h
 *
 *  Created on: Feb 15, 2015
 *      Author: bremer5
 */

#ifndef FUNCTIONHANDLE_H_
#define FUNCTIONHANDLE_H_

#include "FileHandle.h"
#include "GraphHandle.h"
#include "SegmentationHandle.h"

namespace HDFileFormat {

//! A FunctionHandle is a group of handles describing a function
/*! A FunctionHandle describes a function defined through a
 *  DataPoints handle by storing the dimensions of the domain and
 *  the range. A FunctionHandle only accepts GraphHandles and
 *  SegmentationHandles as children
 */
class FunctionHandle : public FileHandle
{
public:

  //! Constructor
  FunctionHandle(HandleType t=H_FUNCTION) : FileHandle(t) {}

  //! Constructor
  FunctionHandle(const char* filename,HandleType t=H_FUNCTION);

  //! Copy constructor
  FunctionHandle(const FunctionHandle& handle);

  //! Destructor
  virtual ~FunctionHandle() {}

  //! Assignment operator
  FunctionHandle& operator=(const FunctionHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new FunctionHandle(*this);}

  //! Add the given handle to the internal data structure but don't write
  virtual FileHandle& add(const FileHandle& handle);

  //! Return the domain
  const std::vector<uint32_t> domain() const {return mDomain;}

  //! Return the range
  uint32_t range() const {return mRange;}

  //! Set the domain
  void domain(const std::vector<uint32_t>& d) {mDomain = d;}

  //! Set the range
  void range(uint32_t r) {mRange = r;}

  //! Return the k-th graph
  HDFileFormat::GraphHandle& getGraph(uint32_t i) {return *this->getChildByType<GraphHandle>(i);}

  //! Get a SegmentationHandle of children by type
  HDFileFormat::SegmentationHandle& getSegmentation(uint32_t i) {return *this->getChildByType<SegmentationHandle>(i);}
protected:

   //! The (sorted) list of indices making up the range
   std::vector<uint32_t> mDomain;

   //! The index of the range
   uint32_t mRange;

   //! Add the local attribute to the node
   virtual int attachXMLInternal(XMLNode& node) const;

   //! Parse the local information from the xml tree
   virtual int parseXMLInternal(const XMLNode& node);


};



}





#endif /* FUNCTIONHANDLE_H_ */
