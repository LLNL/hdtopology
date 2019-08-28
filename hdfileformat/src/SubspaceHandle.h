#ifndef SUBSPACE_HANDLE_H
#define SUBSPACE_HANDLE_H

#include "FileHandle.h"
#include "BasisHandle.h"

#include <vector>

/*
 * SubspaceHandle node will be a chilren of ClusterHandle node
 * Subspace hold all the basis corresponding to each cluster
*/

namespace HDFileFormat{

class SubspaceHandle: public FileHandle
{
public:
  //! The default name of the data set
  static const std::string sDefaultSubspaceName;

  //! Constructor
  explicit SubspaceHandle(HandleType t=H_SUBSPACE);

  //! Constructor
  SubspaceHandle(const char *filename, HandleType t=H_SUBSPACE);

  //! Copy constructor
  SubspaceHandle(const SubspaceHandle& handle);

  //! Destructor
  virtual ~SubspaceHandle();

  //! Assignment operator
  SubspaceHandle &operator=(const SubspaceHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new SubspaceHandle(*this);}

  //! Add the given handle to the internal data structure but don't write
  virtual FileHandle& add(const FileHandle &handle);

  //! Set children basisHandle from subspace handle
  void setChildrenBasis(std::vector<HDFileFormat::Basis> &basis);

  //! Get the baisis object from children by index
  HDFileFormat::Basis& subspaceBasisByIndex(int index);

  //! Get the total subspace number
  uint32_t subspaceNum(){return mSubspaceNum;}

protected:
  //! Parse the xml tree
  virtual int parseXMLInternal(const XMLNode &node);

  //! Add the attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:
  //std::vector<Basis> *mSubspaceBasis;
  uint32_t mSubspaceNum;
};

} //HDFileFormat namespace



#endif
