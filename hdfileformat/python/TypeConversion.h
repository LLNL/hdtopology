/*
 * TypeConversion.h
 *
 *  Created on: Mar 13, 2015
 *      Author: bremer5
 */

#ifndef TYPECONVERSION_H_
#define TYPECONVERSION_H_


const char* numpy_type_name(const std::string c_type_name)
{
  if (c_type_name == "uint32_t")
    return "uint32";
  else if (c_type_name == "int32_t")
    return "int32";
  else if (c_type_name == "float")
    return "float32";
  else if (c_type_name == "double")
    return "float64";
  else
    assert(false);

  return "Unknown";
}

int numpy_type_num(const std::string c_type_name)
{
  if (c_type_name == "uint32_t")
    return NPY_UINT32;
  else if (c_type_name == "int32_t")
    return NPY_INT32;
  else if (c_type_name == "float")
    return NPY_FLOAT32;
  else if (c_type_name == "double")
    return NPY_FLOAT64;
  else
    assert(false);

  return 0;
}


#endif /* TYPECONVERSION_H_ */
