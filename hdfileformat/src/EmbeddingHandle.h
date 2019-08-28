#ifndef EMBEDDING_HANDLE_H
#define EMBEDDING_HANDLE_H

#include "DataBlockHandle.h"

/* EmbeddingHandle group the basis and the actual embedded points
 * it could store one basis node[linear embedding with a basis only], one data node(stored the embedded points)[non-linear embedding], or one basis and one data node[linear embedding with embedded points]
 * 
 * Parameter string pattern "Param1 = value1, Param2 = value2"
*/

namespace HDFileFormat{

enum EmbeddingType{
  EMBEDDING_TYPE_LINEAR_BASIS_ONLY,
  EMBEDDING_TYPE_LINEAR,
  EMBEDDING_TYPE_NONLINEAR,
  EMBEDDING_TYPE_UNDEFINED
};


class EmbeddingHandle: public DataBlockHandle
{
public:
  explicit EmbeddingHandle(HandleType t=H_EMBEDDING);

  //! Constructor
  EmbeddingHandle(const char *filename, HandleType t=H_EMBEDDING);

  //! Copy constructor
  EmbeddingHandle(const EmbeddingHandle&);

  //! Destructor
  virtual ~EmbeddingHandle();
  EmbeddingHandle& operator=(const EmbeddingHandle& handle);
   
  //! Default name
  static const std::string sDefaultEmbeddingName;
  
  //! Clone this handle
  virtual FileHandle* clone() const {return new EmbeddingHandle(*this);}
  
  //! Add the given handle to the internal data structure but don't write  
  virtual FileHandle &add(const FileHandle &handle);
  
  //! Get embedding type
  EmbeddingType getEmbeddingType();

protected:
  //! parse the current node
  virtual int parseXMLInternal(const XMLNode& family);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:
  EmbeddingType mEmbeddingType;
  std::string mEmbeddingParamString;
  //uint32_t mPointCount;
};





}

#endif
