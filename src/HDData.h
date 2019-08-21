#ifndef HDDATA_H
#define HDDATA_H

#include <cstdlib>
#include <vector>
#include <string>
//#include <stdint.h>
#include <typeinfo>
#include <iostream>
#include <stdint.h>
#include <cstdio>

#define LNULL (uint32_t)(-1)

class HDData
{
public:

  //! Default constructor
  HDData() : mData(NULL), mSize(0), mDim(0), mAttr(0), mF(0) {}

  //! Destructor
  ~HDData() {}

  //! Return a pointer to the i-th row
  float* operator[](uint32_t i) {return mData + mAttr*i;}

  //! Return a const pointer to the i-th row
  const float* operator[](uint32_t i) const {return mData + mAttr*i;}

  //! Return the function value of the i'th point
  float f(uint32_t i) const {return *(mData + mAttr*(i) + mF);}//return *(mData + mAttr*(i+1) - 1);}

  //! Compute the distance between two points
  float dist(uint32_t i, uint32_t j) const;

  //! Return the data pointer
  const float* data() const {return mData;}

  //! Return the number of points
  uint32_t size() const {return mSize;}

  //! Return the number of dimensions
  uint32_t dim() const {return mDim;}

  //! Return the number of attrs
  uint32_t attr() const {return mAttr;}

    //! Return the index for function value
  uint32_t func() const {return mF;}


  //! Return the label of the i'th attribute
  const std::string& attribute(uint32_t i) const {return mAttributes[i];}

  //! Get a list of all attribute names
  const std::vector<std::string> attributes() const {return mAttributes;}

  //! Set the data pointer
  void data(float* d) {mData = d;}

  //! Set the size
  void size(uint32_t s) {mSize = s;}

  //! Set the number of dimensions
  void dim(uint32_t d) {mDim = d;}

  //! Set the index for function value
  void func(uint32_t f) {mF = f;}

  //! Set the number of attributes
  void attr(uint32_t d){mAttr = d;}


  void attributes(std::vector<std::string>& attr) {mAttributes = attr;}

private:

  //! A pointer to the 2D array
  float* mData;

  //! The number of rows
  uint32_t mSize;

  //! The number of dimensions
  uint32_t mDim;

  //! The number of attrs
  uint32_t mAttr;

  //! Index for function
  uint32_t mF;

  //! The list of attributes
  std::vector<std::string> mAttributes;
};


class HDFunction : public HDData
{
public:

  //! Default constructor
  HDFunction() : HDData() {};

  //! Destructor
  ~HDFunction() {}

};


#endif
