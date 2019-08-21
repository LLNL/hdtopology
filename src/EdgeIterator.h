/*
 * EdgeIterator.h
 *
 *  Created on: Sep 21, 2018
 *      Author: bremer5
 */

#ifndef EDGEITERATOR_H
#define EDGEITERATOR_H

#include <stdint.h>

//! Virtual Baseclass to define an API to iterate through all edges of a graph
class EdgeIterator
{
public:

  EdgeIterator() {}

  virtual ~EdgeIterator() {}

  //! Advance the iterator
  virtual EdgeIterator& operator++(int i) = 0;

  //! Determine whether we have processed all edges
  virtual bool end() = 0;

  //! Return the pointer to an index pair describing the edge
  virtual const uint32_t* operator*() const = 0;

  //! Determine whether the edges contain the length
  virtual bool hasLength() const {return false;}

  //! Return the current length or 0 is there is no length
  virtual float length() const {return 0;}

  //! Reset the iterator to the beginning
  virtual void reset() = 0;
};

#endif
