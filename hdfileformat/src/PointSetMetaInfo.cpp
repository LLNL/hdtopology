#include "PointSetMetaInfo.h"
#include <string.h>
#include <ctype.h>

namespace HDFileFormat{

#define DEFAULT_STRING_LEN 30

PointSetMetaInfo::PointSetMetaInfo()
  :m_data(NULL),
   m_stringLength(0),
   m_imageSizeX(0),
   m_imageSizeY(0),
   m_channelCount(0),
   m_metaInfoType(META_INFO_UNDFEINED)
{

}

///////////////
/// \brief PointSetMetaInfo::SetStringData
/// \param stringList
///

void PointSetMetaInfo::SetStringData(std::vector<std::string> &stringList)
{
  m_metaInfoType = META_INFO_STRING;
  m_dataTypeSize = sizeof(char);
  m_dataType = std::string("char");

  if(stringList.size() == 0)
    return;
  //find the longest string
  size_t maxStringLength = stringList[0].size();
  for(size_t i=0; i<stringList.size(); i++)
    if(stringList[i].size()>maxStringLength)
      maxStringLength = stringList[i].size();

  if(maxStringLength < DEFAULT_STRING_LEN )
    maxStringLength = DEFAULT_STRING_LEN;

  m_stringLength = maxStringLength;

  //allocate buffer
  m_dataByteSize = m_stringLength*stringList.size();
  m_data = malloc(m_dataByteSize);
  memset(m_data, 1, m_dataByteSize);

  //copy string data
  for(size_t i=0; i<stringList.size(); i++)
  {
    size_t j;
    for(j=0; j<stringList[i].size(); j++)
      ((char*)(m_data))[i*m_stringLength+j] = stringList[i][j];
    //add end of string
    ((char*)(m_data))[i*m_stringLength+j]='\0';
  }
}
/*
std::string trim(const std::string& str,
                 const std::string& whitespace = " \t")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}
*/

//////////
/// \brief PointSetMetaInfo::GetStringData
/// \param stringList
///

void PointSetMetaInfo::GetStringData(std::vector<std::string> &stringList)
{
  if(m_metaInfoType == META_INFO_STRING)
  {
    if(m_stringLength == 0)
      return;
    int sampleCount = m_dataByteSize / m_stringLength;
    for(int i=0; i<sampleCount; i++)
    {
      std::string label;
      for(int j=0; j<m_stringLength; j++)
      {
        char charator = ((char*)(m_data))[i*m_stringLength+j];
        if(isalpha(charator))
          label.push_back( charator );
        //if meet the end of line goto next
        else
        {
          //if(((char*)(m_data))[i*m_stringLength+j]=='\0')
          break;
        }
      }
      stringList.push_back(label);
    }
  }
}




}
