#ifndef FILEDATA_H
#define FILEDATA_H

#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>

#include "Definitions.h"

namespace HDFileFormat {

//! Abstract base class for all data handled by a DataHandle
class FileData
{
public:

  FileData() {}

  virtual ~FileData() {}

  //! Return the current size
  virtual uint32_t size() const = 0;

  //! Write the ascii version of the element to the file
  virtual void writeASCII(std::ofstream& output) const = 0;

  //! Read the ascii version of the element from the file
  virtual void readASCII(std::ifstream& input) = 0;

  //! Write the binary version of the element to the file
  virtual void writeBinary(std::ofstream& output) const = 0;

  //! Read the binary version of the element from the file
  virtual void readBinary(std::ifstream& input) = 0;
};

//! A piece of data encapsulates an array of DataClass
/*! An instance of Data encapsulates an stl vector of DataClass
 *  items alongside the methods to read and write these to a file
 *  stream. Any type or class that overloads << and >> as input
 *  and output operators can be used as DataClass. Note that the
 *  default binary write dumps a flat array.
 */
template <class DataClass>
class Data : public FileData
{
public:

  //! Constructor to use internal array
  Data(uint32_t size=1);

  //! Constructor to use external array
  Data(std::vector<DataClass>* data);

  //! Destructor
  virtual ~Data() {if (mInternalArray) delete mElements;}

  //! Assignment operator (to satisfy windows compiler)
  Data& operator=(const Data& d) {  assert(false); return *this; }

  /*************************************************************************************
   **************************** Array Interface ****************************************
   ************************************************************************************/

  //! Empty the vector
  void clear() {mElements->clear();}

  //! Return the current size
  uint32_t size() const {return mElements->size();}

  //! Resize the array to contain size many elements
  void resize(uint32_t size) {mElements->resize(size);}

  //! Resize the array to contain size many elements
  void resize(uint32_t size, const DataClass& temp) {mElements->resize(size,temp);}

  //! Add one more element at the end of the array
  void push_back(const DataClass& d) {mElements->push_back(d);}

  //! Return the i'th attribute
  DataClass& operator[](uint32_t i) {return mElements->at(i);}

  //! Return the i'th attribute
  const DataClass& operator[](uint32_t i) const {return mElements->at(i);}

  /*************************************************************************************
   **************************** File Interface *****************************************
   ************************************************************************************/

  //! Write the ascii version of the element to the file
  virtual void writeASCII(std::ofstream& output) const;

  //! Read the ascii version of the element from the file
  virtual void readASCII(std::ifstream& input);

  //! Write the binary version of the element to the file
  virtual void writeBinary(std::ofstream& output) const;

  //! Read the binary version of the element from the file
  virtual void readBinary(std::ifstream& input);


protected:

   //! The array of data values
   std::vector<DataClass>* mElements;

private:

   //! Flag to indicate whether we own the array
   const bool mInternalArray;
};

template <class DataClass>
Data<DataClass>::Data(uint32_t size) : mInternalArray(true)
{
  mElements = new std::vector<DataClass>(size);
}

template <class DataClass>
Data<DataClass>::Data(std::vector<DataClass>* data) : mElements(data), mInternalArray(false)
{
  assert (data != NULL);
}

template <class DataClass>
void Data<DataClass>::writeASCII(std::ofstream& output) const
{
  typename std::vector<DataClass>::const_iterator it;

  for (it=mElements->begin();it!=mElements->end();it++)
    output << *it << "\n";
//    it->writeASCII(output);
}
  
template <class DataClass>
void Data<DataClass>::readASCII(std::ifstream& input)
{
  typename std::vector<DataClass>::iterator it;
  
  for (it=mElements->begin();it!=mElements->end();it++) {
    input >> *it;
    //it->readASCII(input);
  }
}

template <class DataClass>
void Data<DataClass>::writeBinary(std::ofstream& output) const
{
  //std::cout << "writing binary data: " << sizeof(DataClass) << " * " << mElements->size() << " = " << sizeof(DataClass) * mElements->size() << std::endl;
  output.write((const char*)&(*mElements)[0],sizeof(DataClass)*mElements->size());
}

template <class DataClass>
void Data<DataClass>::readBinary(std::ifstream& input)
{
  //std::cout << "reading binary data: " << sizeof(DataClass) << " * " << mElements->size() << " = " << sizeof(DataClass) * mElements->size() << std::endl;
  input.read((char*)&(*mElements)[0],sizeof(DataClass)*mElements->size());
}

}

#endif

