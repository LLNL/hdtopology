#include "EmbeddingHandle.h"

namespace HDFileFormat{

/////////// implementation /////////////

const std::string EmbeddingHandle::sDefaultEmbeddingName = "Embedding";

EmbeddingHandle::EmbeddingHandle(HandleType t)
 :DataBlockHandle(t),
  mEmbeddingType(EMBEDDING_TYPE_UNDEFINED)
{
  mID = sDefaultEmbeddingName;
}

EmbeddingHandle::EmbeddingHandle(const char *filename, HandleType t)
 :DataBlockHandle(filename, t),
  mEmbeddingType(EMBEDDING_TYPE_UNDEFINED)
{
  mID = sDefaultEmbeddingName;
}

EmbeddingHandle::EmbeddingHandle(const EmbeddingHandle &handle)
  :DataBlockHandle(handle),
   mEmbeddingType(handle.mEmbeddingType),
   mEmbeddingParamString(handle.mEmbeddingParamString)
{
  
}

EmbeddingHandle::~EmbeddingHandle()
{
}

EmbeddingHandle& EmbeddingHandle::operator=(const EmbeddingHandle& handle)
{
  DataBlockHandle::operator=(handle);
  mEmbeddingType = handle.mEmbeddingType;
  mEmbeddingParamString = handle.mEmbeddingParamString;
  
  return *this;
}

FileHandle& EmbeddingHandle::add(const FileHandle &handle)
{
  switch (handle.type()) {
    case H_BASIS:
      return FileHandle::add(handle);
      break;      
    case H_DATABLOCK:
    case H_SUBSPACE:    
    case H_CLUSTER:
    case H_EMBEDDING:
    case H_COLLECTION:
    default:
      hderror(true,"Nodes of type \"%s\" cannot be nested inside datasets.",handle.typeName());
      return *this;
  }

  return *this;
}

EmbeddingType EmbeddingHandle::getEmbeddingType()
{
  return mEmbeddingType;
}


int EmbeddingHandle::parseXMLInternal(const XMLNode &node)
{
  DataBlockHandle::parseXMLInternal(node);

  uint32_t type;
  getAttribute(node, "embeddingType", type);
  mEmbeddingType = static_cast<EmbeddingType>(type);
  getAttribute(node, "EmbeddingParameters", mEmbeddingParamString);
  
  return 1;
}

int EmbeddingHandle::attachXMLInternal(XMLNode& node) const
{
  DataBlockHandle::attachXMLInternal(node);
  
  uint32_t type = static_cast<uint32_t>(mEmbeddingType);
  addAttribute(node, "embeddingType", type);
  addAttribute(node, "EmbeddingParameters", mEmbeddingParamString);

  return 1;
}

}
