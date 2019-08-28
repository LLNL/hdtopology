#ifndef PARAMETER_PARSER_H
#define PARAMETER_PARSER_H

#include <vector>
#include <string>
#include <utility>

/*
 * Provide the functionality to parse the modify parameter strings
 *
 * Why use string to store the parameters instead of in the same XML?
 * A: In the library, the XML read/write is fix for the ClusterHandle, which can hold cluster
 * generate from all kinds of clustering algorithm with various key of parameter. There is
 * no uniform way to address together in the ClusterHandle. In other word, we don't know what
 * kind of parameter stored in the XML when we read the file. Using XML require the program to know what
 * field to read and write.
 *
 * (the string can be a XML format, a XML embed in a XML?)
 *
 * Author: Shusen Liu  Date: Oct 15, 2014
*/

namespace HDFileFormat{

class ParameterParser
{
public:
  //! static methods
  static std::vector<std::pair<std::string, std::string> > ParseParameterToStringParis(std::string);
  static void AppendParameter(std::string& paramString, std::string param, std::string value);
  static void InsertParameter(std::string& paramString, std::string param, std::string value);

  //! member function
  void setParameterString(std::string& params)
  {
    mParameterString = params;
    mParameterPairs = ParseParameterToStringParis(params);
  }

private:
  std::string mParameterString;
  std::vector<std::pair<std::string, std::string> > mParameterPairs;
};

}


#endif
