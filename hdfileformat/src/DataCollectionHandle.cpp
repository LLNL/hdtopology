/*
 * DataCollectionHandle.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: bremer5
 */

#include <cstdlib>
#include <string>
#include <iostream>
#include <errno.h>

#include "DataCollectionHandle.h"

namespace HDFileFormat {

const uint16_t DataCollectionHandle::sMajorVersion = 1;

const uint16_t DataCollectionHandle::sMinorVersion = 0;

const uint8_t DataCollectionHandle::sPrecision = 8;

const std::string DataCollectionHandle::sDefaultName = "Collection";

DataCollectionHandle::DataCollectionHandle()
  :FileHandle(H_COLLECTION),
   mCollection(sDefaultName),
   mMajor(sMajorVersion),
   mMinor(sMinorVersion)
{
  mID = sDefaultName;
}

DataCollectionHandle::DataCollectionHandle(const std::string& filename)
  :FileHandle(filename.c_str(),H_COLLECTION),
   mCollection(sDefaultName),
   mMajor(sMajorVersion),
   mMinor(sMinorVersion)
{
  mID = sDefaultName;
}

DataCollectionHandle::DataCollectionHandle(const char* filename) :
    FileHandle(filename,H_COLLECTION), mCollection(sDefaultName),
    mMajor(sMajorVersion), mMinor(sMinorVersion)
{
  mID = sDefaultName;
}

DataCollectionHandle::DataCollectionHandle(const DataCollectionHandle& handle)
  :FileHandle(handle),
   mCollection(handle.mCollection),
   mMajor(handle.mMajor),
   mMinor(handle.mMinor)
{
}

DataCollectionHandle::~DataCollectionHandle()
{
}

DataCollectionHandle& DataCollectionHandle::operator=(const DataCollectionHandle& handle)
{
  FileHandle::operator=(handle);

  mCollection = handle.mCollection;

  return *this;
}

int DataCollectionHandle::attach(const char* filename)
{
  FileOffsetType header_size;
  char* header;
  std::ifstream file;

  // Remove all the old information
  clear();

  mFileName = std::string(filename);

  if (!openInputFile(mFileName,file, std::ios_base::binary))
    return 0;

  // While the step of using i seems completely useless things
  // will actually produce a run-time error when we directly pass
  // -sizeof() to the seekg function
  int i = -int(sizeof(FileOffsetType)); //minus apply to unsigned will still be unsigned
  file.seekg(i,std::ios_base::end);
  //file.seekg(-(sizeof(FileOffsetType)),std::ios_base::end);

  if (file.fail())
    return 0;

  file.read((char *)&this->mOffset,sizeof(FileOffsetType));

  if (file.fail())
    return 0;

  // Allocate a buffer to read the header
  header_size = static_cast<FileOffsetType>((FileOffsetType)file.tellg() - this->mOffset - sizeof(FileOffsetType));
  header = new char[header_size+1];

  // Point the file stream to the start of the header
  if (!rewind(file))
    return 0;

  // Read the header
  file.read((char *)header,header_size);


  if (file.fail())
    return 0;

  // Close the file
  file.close();

  // Make the header into a valid string
  header[header_size] = '\0';

  // fprintf(stderr,"Header Offset: %lld\n", mOffset);
  // fprintf(stderr,"Header Offset: %lld\n%s##########", mOffset, header);

  XMLNode root;
  XMLCSTR tag = NULL;
  XMLResults *results = NULL;


  root = XMLNode::parseString(header,tag,results);

  if (results != NULL) {
    delete[] header;
    return 0;
  }

  std::string str = root.getName();
  std::string str1 = typeName();
  hderror((strcmp(root.getName(),typeName()) != 0),"The topmost node of a topology file should be a collection node.");

  parseXML(root);

  delete[] header;

  return 1;
}

void DataCollectionHandle::write(const char* filename)
{
  std::vector<DatasetHandle>::iterator dIt;

 if (filename != NULL)
    this->mFileName = std::string(filename);
  else {
    hderror(this->mFileName==sEmptyString,"No internal file name set. Need a file name to write to");
  }

  std::ofstream file;
  openOutputFile(this->mFileName,file,true); // For now we are not using std::ios::binary

  // Make sure that our ascii values have the desired precision
  file.precision(sPrecision);

  // Make sure that we use scientific notation
  std::scientific(file);

  //! Now write all datasets of this file
  for (dIt=mDatasets.begin();dIt!=mDatasets.end();dIt++)
    (*dIt).writeData(file,this->mFileName);

  // Rewrite the xml-footer with the new info included
  attachXMLFooter(file);

  file.close();

}

FileHandle& DataCollectionHandle::add(const FileHandle& handle)
{
  switch (handle.type()) {
    case H_DATASET:
      //why cast to reference??
      mDatasets.push_back(dynamic_cast<const DatasetHandle&>(handle));
      //mDatasets.push_back(static_cast<const Dataset>handle);
      mDatasets.back().topHandle(this);
      return mDatasets.back();
      break;
    default:
      hderror(true,"Unknown handle type, cannot attach a %d to a clan",handle.type());
      break;
  }

  // Note that this is a bogus return to satisfy the compiler. You should hit
  // the error before coming to here
  return mDatasets.back();
}

void DataCollectionHandle::updateMetaData(const char* filename)
{
  if (filename != NULL)
    this->mFileName = std::string(filename);
  else {
    hderror(this->mFileName==sEmptyString,"No internal file name set. Need a file name to write to");
  }

  std::ofstream file(this->mFileName.c_str(),std::ios::in | std::ios::out);
  hderror(file.fail(),"Could not open file \"%s\" with mode \"%s\". Got errno %d = \"%s\".\n",this->mFileName.c_str(),
          std::ios::in | std::ios::out,errno,strerror(errno));

  // Point the file to the current start of the header
  file.seekp(mOffset,std::ios_base::beg);

  // Rewrite the xml-footer with the new info included
  attachXMLFooter(file);

  file.close();
}


void DataCollectionHandle::clear()
{
  FileHandle::clear();

  mCollection = sDefaultName;
  mDatasets.clear();
  mMajor = sMajorVersion;
  mMinor = sMinorVersion;
}

void DataCollectionHandle::appendData(FileHandle& handle)
{
  fprintf(stderr,"DataCollectionHandle::appendData of handle \"%s\"\n",handle.typeName());

  if (mFileName == sEmptyString)
    fprintf(stderr,"Cannot append data to file since ClanHandle is not attached yet.");

  std::ofstream file;
  if (handle.encoding())
    openOutputFileToAppend(this->mFileName,file,false);
  else
    openOutputFileToAppend(this->mFileName,file,true);


  // Make sure that our ascii values have the desired precision
  file.precision(sPrecision);

  // Make sure that we use scientific notation
  std::scientific(file);

  // Point the file to the current start of the header
  file.seekp(mOffset,std::ios_base::beg);

  // Write the data of the new handle. This implicitly update the offset values
  // This will make sure all the children's data is write recursively
  handle.writeData(file,this->mFileName);

  // Rewrite the xml-footer with the new info included
  attachXMLFooter(file);

  file.close();
}

int DataCollectionHandle::parseXML(const XMLNode& node)
{
  FileHandle* handle;

  fprintf(stderr,"DataCollectionHandle::attachXMLInternal \"%s\"\n",this->typeName());

  parseXMLInternal(node);

  for (int i=0;i<node.nChildNode();i++) {

    handle = this->constructHandle(node.getChildNode(i).getName(),mFileName);
    handle->topHandle(this);

    switch (handle->type()) {
      case H_DATASET:
        handle->parseXML(node.getChildNode(i));
        mDatasets.push_back(*dynamic_cast<DatasetHandle*>(handle));
        break;
      default:
        hderror(true,"Invalid xml structure. A collection should not have a child of type %d",handle->type());
        break;
    }
    delete handle;
  }
  return 1;
}

int DataCollectionHandle::parseXMLInternal(const XMLNode& node)
{
  FileHandle::parseXMLInternal(node);

  if (node.getAttribute("major",0) == NULL)
    fprintf(stderr,"Could not find required \"major\" attribute for file handle.\n");
  else {
    mMajor = (uint16_t) atoi(node.getAttribute("minor",0));
  }

  if (node.getAttribute("minor",0) == NULL)
    fprintf(stderr,"Could not find required \"minor\" attribute for file handle.\n");
  else {
    mMinor = (uint16_t) atoi(node.getAttribute("minor",0));
  }

  hderror(mMajor > sMajorVersion,"Version number missmatch. File needs version %d.%d but code is version %d.%d",mMajor,mMinor,
          sMajorVersion,sMinorVersion);

  hderror((mMajor==sMajorVersion) && (mMinor > sMinorVersion),"Version number missmatch. File needs version %d.%d but code is version %d.%d",mMajor,mMinor,
          sMajorVersion,sMinorVersion);


  if (node.getAttribute("name",0) == NULL)
    mCollection = sDefaultName;
  else {
    mCollection = std::string(node.getAttribute("collection",0));
  }

  return 1;
}

int DataCollectionHandle::attachXML(XMLNode &node) const
{
  //attach internal
  this->attachXMLInternal(node);

  //attach the tree
  XMLNode child;
  for (uint8_t i=0;i<mDatasets.size();i++) {

    child = node.addChild(mDatasets[i].typeName());
    mDatasets[i].attachXML(child);
  }

  return 1;
}



int DataCollectionHandle::attachXMLInternal(XMLNode& node) const
{
  FileHandle::attachXMLInternal(node);

  addAttribute(node,"major",mMajor);
  addAttribute(node,"minor",mMinor);

  addAttribute(node,"collection",mCollection.c_str());

  return 1;
}

int DataCollectionHandle::attachXMLFooter(std::ofstream& file)
{
  // Record the offset as start of the header
  mOffset = file.tellp();

  // Then we create the xml hierarchy
  XMLNode clan;

  clan = XMLNode::createXMLTopNode(typeName());
  attachXML(clan);

  // Create the corresponding header string
  XMLSTR header = clan.createXMLString();

  file << header;


  FileOffsetType o = static_cast<uint64_t>(mOffset);
  fprintf(stderr,"\n header offset: %llu\n",o);
  // fprintf(stderr,"\n header offset: %llu\n%s#########\n",o,header);
  fprintf(stderr,"filesize: %llu\n",static_cast<uint64_t>(file.tellp()));
  file.write((const char*)&o,sizeof(FileOffsetType));

  return 1;
}


}
