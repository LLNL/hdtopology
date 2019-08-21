/*
 * Flags.h
 *
 *  Created on: Jan 29, 2019
 *      Author: bremer5
 */

#ifndef FLAGS_H
#define FLAGS_H

#include <cstdlib>
#include <stdint.h>


class Flags
{
public:

  //! Default constructor
  Flags(const uint8_t* flags=NULL) : mFlags(flags) {}

  //! Copy constructor
  Flags(const Flags& f) : mFlags(f.mFlags) {}

  //! Destructor
 ~Flags() {}

 //! Return the ith flag
 bool operator[](uint32_t i) const {if (mFlags==NULL) return true;else return mFlags[i];}

//private:
  const uint8_t* mFlags;
};



#endif




