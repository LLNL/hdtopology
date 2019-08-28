#ifndef HISTOGRAM_HANDLE_H
#define HISTOGRAM_HANDLE_H

#include "DataBlockHandle.h"

#include <iostream>

namespace HDFileFormat{

/////////////////// Distribution Handle /////////////////////////

class HistogramHandle: public DataBlockHandle
{
public:
  //! The default name of the data set
  static const std::string sDefaultBasisName;

  //! Constructor
  explicit HistogramHandle(HandleType t=H_HISTOGRAM);

  //! Constructor
  HistogramHandle(const char *filename, HandleType t=H_HISTOGRAM);

  //! Copy constructor
  HistogramHandle(const HistogramHandle&);

  //! Destructor
  virtual ~HistogramHandle();

  //! Assignment operator
  HistogramHandle& operator=(const HistogramHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new HistogramHandle(*this);}

  //! add children
  virtual FileHandle& add(const FileHandle &handle);


protected:
  //! Parse the xml tree
  virtual int parseXMLInternal(const XMLNode &node);

  //! Add the attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:


};



}

#endif
