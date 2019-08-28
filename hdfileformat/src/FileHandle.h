#ifndef FILEHANDLE_H
#define FILEHANDLE_H

#include <stdio.h> //need this on linux
#include <string>
#include <sstream>
#include <vector>

#include "Definitions.h"
#include "xmlParser.h"
#include <fstream>

namespace HDFileFormat {

/*!
*
* The handle system provide a lazy load scheme, the data will only be load into memory when data acess is requested.
* It hold the reference to the data location
*
* [Possible Issue:] - what if I save the file, during the file is still opened, the handle may be came invalid
* [Solution:] - Put the XML head at the end of the file, and always append the new data at the end of the file
* before the header, as a result, all the previous index will not need be updated
*
*   Created on : Aug 19, 2014
*      Authors : bremer5, Shusen Liu(shusenl)
*
*/


#define HANDLE_COUNT 17

enum HandleType {
  H_COLLECTION = 0,
  H_DATASET = 1,

  H_DATABLOCK = 2,
  H_DATAPOINTS =3,
  H_CLUSTER = 4,
  H_EMBEDDING = 5,
  H_GRAPH = 6,
  H_SEGMENTATION = 7,
  H_FUNCTION = 8,

  H_HIERARCHY = 9,
  H_SUBSPACE = 10,
  H_BASIS = 11,

  H_VOLSEGMENT = 12,
  H_DATAPOINTS_METAINFO = 13,
  H_HISTOGRAM = 14,
  H_DISTRIBUTION = 15,

  H_UNDEFINED = 16
};

//! Declarations
class DataPointsHandle;
class DataBlockHandle;
class GraphHandle;
class ClusterHandle;
class SubspaceHandle;
class BasisHandle;

//! The common baseclass for all file handles
class FileHandle
{
public:

  //! Convinience variable defining an empty string
  static const std::string sEmptyString;

  //! Default seperator used for multiple string arguments
  static const char sStringSeperator;

  //! Return the name of the given handle type
  static const char* typeName(HandleType t);

  //! Friend declaration to allow access to the hidden parsing functions
  friend class DataCollectionHandle;

  //! Friend declaration to allow access to the hidden parsing functions
  friend class SegmentationHandle;

  //! Friend declaration to allow access to the hidden parsing functions
  friend class ClusterHandle;

  //! Friend declaration to allow access to the hidden parsing functions
  friend class DatasetHandle;


  //! Static function to create various types of handles
  static FileHandle* constructHandle(const char* typeName, const char* filename);

  //! Static function to create various types of handles
  static FileHandle* constructHandle(const char* typeName, const std::string& filename);

  //! Constructor, single argument constructor using explicit to avoid unwanted type converstion from HandleType to FileHandle
  explicit FileHandle(HandleType t=H_UNDEFINED);

  //! Constructor
  FileHandle(const char* filename,HandleType t=H_UNDEFINED);

  //! Copy constructor
  FileHandle(const FileHandle& handle);

  //! Destructor
  virtual ~FileHandle();

  //! check if handle are valid
  virtual bool isValid()
  {return (mTopHandle!=NULL) && mFileName.size();}
  //{return  mFileName.size();}
  //{return  (mTopHandle!=NULL);}


  //! Assignment operator
  FileHandle& operator=(const FileHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const = 0;

  //! instantiate -  load the mDataBuffer by reading from the file
  virtual void instantiate();

  //! Return the handle's type
  HandleType type() const { return mType;}

  //! Return the string identifying this handle's type
  const char* typeName() const;

  //! Update handle specific ID name for all type of handle
  virtual void idString(const std::string ID){mID = ID;}

  //! Return handle specific ID name for all type of handle
  virtual std::string idString(){return mID;}

  //! Update Meta Info String 
  virtual void metaInfoString(const std::string metaInfo){mMetaInfo = metaInfo;}

  //! Return Meta Info String
  virtual std::string metaInfoString(){return mMetaInfo;} 

  //! Return whether this handle is ascii or binary
  bool encoding() {return mASCIIFlag;}

  //! Switch the encoding type
  void encoding(bool ascii_flag) {mASCIIFlag=ascii_flag;}

  //! Append the given handle and re-write the file
  void append(FileHandle& handle)
  {
    FileHandle& h = add(handle);
    topHandle()->appendData(h);
  }

  //! Add the given handle to the internal data structure but don't write
  virtual FileHandle& add(const FileHandle& handle);

  // Children related
  uint32_t childCount() const {return mChildren.size();}

  //! Return the i'th data points handle (for the python interface)
  HDFileFormat::DataPointsHandle& getDataPoints(uint32_t i);

  //! Return the i'th data block handle (for the python interface)
  HDFileFormat::DataBlockHandle& getDataBlock(uint32_t i);

  //! Return the i'th graph handle (for the python interface)
  HDFileFormat::GraphHandle& getGraph(uint32_t i);

  //! Return the i'th cluster handle (for the python interface)
  HDFileFormat::ClusterHandle& getCluster(uint32_t i);

  //! Return the i'th subspace handle (for the python interface)
  HDFileFormat::SubspaceHandle& getSubspace(uint32_t i);

  //! Return the i'th basis handle (for the python interface)
  HDFileFormat::BasisHandle& getBasis(uint32_t i);

  //! Return the cluster handle by string (for the python interface)
  HDFileFormat::ClusterHandle& getClusterByName(std::string clusterName);

  //! Can't modify the entry through these access
  //! Get child node by type, pointer is easier to initialize and check than reference
  template<typename derivedType>
  void getFirstChildByType( derivedType& );

  template<typename ChildType>
  ChildType* getChildByType(uint32_t i);


  //! Get child node by type and id
  template<typename HandleType>
  HandleType* getChildByType(std::string IDstring);

  //! Get child node by type and id resursively
  template<typename HandleType>
  HandleType* getChildInFullHierarchyByType(std::string IDstring);

  //! Get children nodes by type
  template<typename derivedType>
  void getAllChildrenByType( std::vector<derivedType> &);

  //! Get children number by type string
  int getChildrenCountByType(const char* typeName);

  //! Get children number by type
  int getChildrenCountByType(HandleType type);

  //! Get children number by type
  template<typename HandleType>
  int getChildrenCountByType();

  //! Get all children
  std::vector<FileHandle*> getAllChildren(){return this->mChildren;}

protected:
  //! The collection of child nodes
  std::vector<FileHandle*> mChildren;

  //! The ID for this node, will the be ID name for different handle
  std::string mID;

  //! meta info string (meta Information store as json)
  std::string mMetaInfo;

  //! The type of this handle
  const HandleType mType;

  //! The filename we are connected to
  std::string mFileName;

  //! The file offset of the data
  std::streamoff mOffset;

  //! Pointer to the top most handle. Careful, special rules apply in the copy constructor or "="
  FileHandle* mTopHandle;

  //! Flag to indicate whether we should encode binary=false or ascii=true
  bool mASCIIFlag;

  //! Reset all values to their default uninitialized values
  void clear();

  //! Return the top handle
  const FileHandle* topHandle() const {return mTopHandle;}

  //! Return the top handle
  FileHandle* topHandle() {return mTopHandle;}

  //! Set the top handle
  virtual void topHandle(FileHandle* top) {mTopHandle = top;}

  //! Append the data of the given handle to the file and re-write the footer
  virtual void appendData(FileHandle& handle);

  //! Allow writing data to file (in this case children, DataBlockHandle need to call this)
  virtual int writeData(std::ofstream& output,const std::string& filename);

  //! Writing per-handle data to file
  virtual int writeDataInternal(std::ofstream& output, const std::string& filename){return 1;}

  //! Write serialization of metaInfo (the class should provide serialization function)
  virtual int writeSerializeData(std::ofstream& output) {return 1;}

  //! Parse the xml tree
  virtual int parseXML(const XMLNode& node);

  //! Parse the local information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Attach the xml tree
  virtual int attachXML(XMLNode& node) const;

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

  //! Add objet serialization
  bool serializeObject();

  //! Set the file pointer to the correct offset
  int rewind(std::ifstream& file) const;

  //! Open the given file for output in the mode provided
  int openOutputFile(const std::string& filename, std::ofstream& file,
                     bool binary) const;

  //! Open the given file for output in the mode provided
  int openOutputFile(const char* filename, std::ofstream& file,
                     bool binary) const;

  //! Open the given file for output in the mode provided
  int openOutputFileToAppend(const std::string& filename, std::ofstream& file,
                             bool binary) const;

  //! Open the given file for output in the mode provided
  int openOutputFileToAppend(const char* filename, std::ofstream& file,
                             bool binary) const;

  //! Open the given file for input in the mode provided
  int openInputFile(const std::string& filename, std::ifstream& file,
                    bool binary) const;

  //! Open the given file for input in the mode provided
  int openInputFile(const char* filename, std::ifstream& file,
                    bool binary) const;

  //! Add a simple attribute to an xml node
  template <typename AttrType>
  void addAttribute(XMLNode& node,const char* attr_name, AttrType attr) const;

  //! Get a simple Attribute
  template <typename AttrType>
  void getAttribute(const XMLNode& node,const char* attr_name, AttrType& attr);

  //! Convinience function to split string attributes
  std::vector<std::string> splitString(const std::string& input,const char sep = sStringSeperator);

  //! Convinience function to split string to numbers
  template<typename T>
  std::vector<T> splitStringToNumber(const std::string& input,const char sep = sStringSeperator);

};

template<typename HandleClass>
HandleClass* FileHandle::getChildByType(uint32_t child_index)
{
  HandleClass handle;
  uint32_t count = 0;

  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type() == handle.type()) {

      if (count == child_index)
        return dynamic_cast<HandleClass*>(mChildren[i]);

      count++;
    }

  return NULL;
}

template<typename HandleClass>
HandleClass* FileHandle::getChildByType(std::string id)
{
  HandleClass handle;

  for(size_t i=0; i<mChildren.size(); i++) {
    if(mChildren[i]->type() == handle.type() && (mChildren[i]->idString() == id))
        return dynamic_cast<HandleClass*>(mChildren[i]);

  }

  return NULL;
}

template<typename HandleClass>
HandleClass *FileHandle::getChildInFullHierarchyByType(std::string id)
{
  HandleClass handle;

  for(size_t i=0; i<mChildren.size(); i++)
  {
    if(mChildren[i]->type() == handle.type() && (mChildren[i]->idString() == id))
        return dynamic_cast<HandleClass*>(mChildren[i]);
  }

  for(size_t i=0; i<mChildren.size(); i++)
  {
    HandleClass* handleFound = mChildren[i]->getChildInFullHierarchyByType<HandleClass>(id);
    if( handleFound ) //if return not null
      return handleFound;
  }

  return NULL;
}

template<class HandleClass>
void FileHandle::getAllChildrenByType(std::vector<HandleClass> &childrenVec)
{
  HandleClass handle;

  for(size_t i=0; i<mChildren.size(); i++)
    if(mChildren[i]->type() == handle.type())
      childrenVec.push_back( *dynamic_cast<HandleClass*>(mChildren[i]->clone()));
}


template<class HandleClass>
int FileHandle::getChildrenCountByType()
{
  HandleClass handle;

  int childrenCount = 0;
  for (int32_t i=0;i<mChildren.size();i++)
    if(handle.type() == mChildren[i]->type())
      childrenCount++;
  return childrenCount;

}


template <typename AttrType>
void FileHandle::addAttribute(XMLNode& node,const char* attr_name, AttrType attr) const
{
  std::stringstream output;

  output << attr;
  output.flush();
  node.addAttribute(attr_name,output.str().c_str());
}

//! There is a special implementation for std::string in the cpp file

template <typename AttrType>
void FileHandle::getAttribute(const XMLNode& node,const char* attr_name, AttrType& attr)
{
  if (node.getAttribute(attr_name,0) == NULL)
    fprintf(stderr,"Could not find \"%s\" attribute for file handle.\n",attr_name);
  else {
    std::stringstream input(node.getAttribute(attr_name));

    input >> attr;
  }
}

// make stream work with enum
//inline std::istream & operator>>(std::istream & str,  HierarchyType v) {
//  unsigned int type = 0;
//  if (str >> type)
//    v = static_cast<HierarchyType>(type);
//  return str;
//}

template <>
inline void FileHandle::getAttribute<std::string>(const XMLNode& node,const char* attr_name, std::string& attr)
{
  if (node.getAttribute(attr_name,0) == NULL)
    fprintf(stderr,"Could not find \"%s\" attribute for file handle.\n",attr_name);
  else {
    attr = node.getAttribute(attr_name);
  }
}

template<typename T>
std::vector<T> FileHandle::splitStringToNumber(const std::string& input,const char sep)
{
  std::vector<T> tokens;
  std::istringstream str(input);
  std::string token;

  while(getline(str, token, sep))
  {
    //convert token to the T type
    std::istringstream ss(token);
    T result;
    ss >> result;
    tokens.push_back(result);
  }

  return tokens;
}


}

#endif
