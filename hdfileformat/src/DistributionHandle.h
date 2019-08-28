#ifndef DISTRIBUTION_HANDLE_H
#define DISTRIBUTION_HANDLE_H

#include "DataBlockHandle.h"

namespace HDFileFormat{

/////////////////// Distribution Handle /////////////////////////

class DistributionHandle: public DataBlockHandle
{
public:
  //! The default name of the data set
  static const std::string sDefaultName;

  //! Constructor
  explicit DistributionHandle(HandleType t=H_DISTRIBUTION);

  //! Constructor
  DistributionHandle(const char *filename, HandleType t=H_DISTRIBUTION);

  //! Copy constructor
  DistributionHandle(const DistributionHandle&);

  //! Destructor
  virtual ~DistributionHandle();

  //! Assignment operator
  DistributionHandle& operator=(const DistributionHandle& handle);

  //! Clone this handle
  virtual DistributionHandle* clone() const {return new DistributionHandle(*this);}

protected:
  //! Parse the xml tree
  virtual int parseXMLInternal(const XMLNode &node);

  //! Add the attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:


};



}

#endif
