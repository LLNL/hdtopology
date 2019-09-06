#ifndef NEIGHBORHOOD_H
#define NEIGHBORHOOD_H

#include <cstdlib>
#include <vector>
#include <cstdint>

class Neighborhood
{
public:

  Neighborhood() : mData(NULL), mSize(0), mLength(NULL) {}
  Neighborhood(uint32_t* data, uint32_t s) : mData(data), mSize(s), mLength(NULL) {}
  ~Neighborhood() {}

  uint32_t* operator[](uint32_t i) {return mData + 2*i;}
  const uint32_t* operator[](uint32_t i) const {return mData + 2*i;}

  uint32_t e(uint32_t i, uint32_t k) const {return mData[2*i + k];}

  uint32_t size() const {return mSize;}

  bool hasLength() const {return (mLength != NULL);}

  float length(uint32_t i) const {
    if(mLength)
      return mLength[i];
    else
      return 0;
  }

  void data(uint32_t* d) {mData = d;}
  void size(uint32_t s) {mSize = s;}
  void setLength(float* l) {mLength = l;}

  uint32_t load_neighborhood(const char* filename);

private:

  uint32_t *mData;
  uint32_t mSize;
  float *mLength;
};


#endif
