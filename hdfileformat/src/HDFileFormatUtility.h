#ifndef HD_FILE_FORMAT_UTILITY_H
#define HD_FILE_FORMAT_UTILITY_H

//convert a type to string at runtime
#ifndef _WIN32
#include <cxxabi.h>
#include <string>

#include <typeinfo>
#include <exception>

#endif

namespace HDFileFormat{

template<typename T>
inline const char* identifyTypeByPointer(T toBeIdentified)
{
#ifndef _WIN32
  int status;
  const char *typeName = abi::__cxa_demangle(typeid(*toBeIdentified).name(), 0, 0, &status);
#else
  const char *typeName = (const char*)typeid(*toBeIdentified).name();
  if (std::string(typeName).substr(0, 6) == std::string("class "))
    return typeName + 6;//skip the first 6 char
#endif
  return typeName;
}

}

#endif
