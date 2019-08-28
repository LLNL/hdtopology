#ifndef BASIS_HANDLE_H
#define BASIS_HANDLE_H

#include "DataBlockHandle.h"
#include "Basis.h"

#include <iostream>

namespace HDFileFormat{

/////////////////// Basis Handle /////////////////////////

class BasisHandle: public DataBlockHandle
{
public:
  //! The default name of the data set
  static const std::string sDefaultBasisName;

  //! Constructor
  explicit BasisHandle(HandleType t=H_BASIS);

  //! Constructor
  BasisHandle(const char *filename, HandleType t=H_BASIS);

  //! Copy constructor
  BasisHandle(const BasisHandle&);

  //! Destructor
  virtual ~BasisHandle();

  //! Assignment operator
  BasisHandle& operator=(const BasisHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new BasisHandle(*this);}

  //! add children
  virtual FileHandle& add(const FileHandle &handle);

  void setBasis(HDFileFormat::Basis &basis);

  HDFileFormat::Basis &basis();

  int basisDimension(){return mDimension;}

protected:
  //! Parse the xml tree
  virtual int parseXMLInternal(const XMLNode &node);

  //! Add the attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:

  HDFileFormat::Basis mBasis;

};



}

#endif
