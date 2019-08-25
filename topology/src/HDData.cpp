#include <cmath>
#include "HDData.h"


float HDData::dist(uint32_t i, uint32_t j) const
{
  float mag = 0;

  for (uint32_t k=0;k<mDim;k++) 
    mag += pow((*this)[j][k] - (*this)[i][k],2);

  mag = sqrt(mag);
  return mag;
}
