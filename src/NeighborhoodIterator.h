/*
 * NeighborhoodIterator.h
 *
 *  Created on: Sep 21, 2018
 *      Author: bremer5
 */

#ifndef NEIGHBORHOODITERATOR_H
#define NEIGHBORHOODITERATOR_H

#include "EdgeIterator.h"
#include "Neighborhood.h"

//! Implementation of an iterator using a Neighborhood as source
class NeighborhoodIterator : public EdgeIterator
{
public:

  //! Constructor
  NeighborhoodIterator(const Neighborhood& neighborhood) : mNeighborhood(neighborhood),
  mHasLength(neighborhood.hasLength()),mIndex(0) {}

  //! Destructor
  ~NeighborhoodIterator() {}

  //! Advance the iterator
  EdgeIterator& operator++(int i) {mIndex++;return *this;}

  //! Determine whether we have processed all edges
  bool end() {return (mIndex >= mNeighborhood.size());}

  //! Return the pointer to an index pair describing the edge
  const uint32_t* operator*() const {return mNeighborhood[mIndex];}

  //! Determine whether the edges contain the length
  bool hasLength() const {return mHasLength;}

  //! Return the current length or 0 is there is no length
  float length() const {return mNeighborhood.length(mIndex);}

  //! Reset the iterator to the beginning
  void reset() {mIndex = 0;}

private:

  const Neighborhood& mNeighborhood;

  const bool mHasLength;

  //! The currently running index
  uint64_t mIndex;
};

#endif
