/*
 * Definitions.h
 *
 *  Created on: Aug 19, 2014
 *      Author: bremer5
 */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#ifdef _WIN32
  #include <stdint.h>
#else
//not included in visual studio 2012, C99 header
  #include <stdint.h>
  //#include <inttypes.h>
#endif

#include <assert.h>

namespace HDFileFormat {

typedef uint64_t FileOffsetType;

#if defined(WIN32) || defined(WIN64)

#define hdwarning(msg,...) {;}
#define hderror(condition,msg,...) {;}
#define hdmessage(condition,msg,...) {;}

#else

#ifndef NDEBUG

#include <stdio.h>
#include <string.h> //<cstring> conflict with std::string namespace on linux

#define hdwarning(msg,...) {char error[200] = "WARNING: %s::%u:\n\t";strcat(error,msg);strcat(error,"\n");fprintf(stderr,error,__FILE__,__LINE__ , ## __VA_ARGS__);}
#define hderror(condition,msg,...) {if ((condition)) { char error[200] = "ERROR: %s::%u:\n\t";strcat(error,msg);strcat(error,"\n");fprintf(stderr,error,__FILE__,__LINE__ , ## __VA_ARGS__);assert(false);}}
#define hdmessage(condition,msg,...)  {if ((condition)) { char error[200] = "WARNING: %s::%u:\n";strcat(error,msg);strcat(error,"\n");fprintf(stderr,error,__FILE__,__LINE__ , ## __VA_ARGS__);}}

#else

#define hdwarning(msg,...) {;}
#define hderror(condition,msg,...) {;}
#define hdmessage(condition,msg,...)  {;}

#endif

#endif

}

#endif /* DEFINITIONS_H_ */
