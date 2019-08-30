#include <cstdio>
#include <fstream>
#include <cassert>
#include <string>
#include <cstring> //for memcpy decleration
#include "Neighborhood.h"

uint32_t Neighborhood::load_neighborhood(const char* filename)
{
  std::vector<uint32_t> edges;
  std::vector<float> length;

  std::ifstream fin(filename);
  //FILE* input = fopen(filename,"r");

  uint32_t e[2];
  //char* line = NULL;
  size_t linecap = 0;
  float f;
  std::string strline;
  
  std::getline(fin, strline);
  const char* line = strline.c_str();

  //getline(&line, &linecap, input); //POSIX only

  if (sscanf(line,"%d %d %f",e,e+1,&f) == 2) {
    edges.push_back(e[0]);
    edges.push_back(e[1]);

	while (fin >> e[0] && fin >> e[1]) {
		edges.push_back(e[0]);
		edges.push_back(e[1]);
	}
    //while (fscanf(input,"%d %d", e,e+1) != EOF) {
    //  edges.push_back(e[0]);
    //  edges.push_back(e[1]);
    //}
  }
  else {
    edges.push_back(e[0]);
    edges.push_back(e[1]);
    length.push_back(f);

	while (fin >> e[0] && fin >> e[1] && fin >> f) {
		edges.push_back(e[0]);
		edges.push_back(e[1]);
		length.push_back(f);
	}
    //while (fscanf(input,"%d %d %f", e,e+1,&f) != EOF) {
    //  edges.push_back(e[0]);
    //  edges.push_back(e[1]);
    //  length.push_back(f);
    //}
  }
      
  //fclose(input);
  fin.close();

  if (!length.empty())
    assert (edges.size() == 2*length.size());

  mSize = edges.size()/2;

  if (mData != NULL)
    delete[] mData;

  mData = new uint32_t[2*mSize];
  memcpy(mData,&edges[0],2*mSize*sizeof(uint32_t));

  if (mLength != NULL) {
    delete[] mLength;
    mLength = NULL;
  }

  if (!length.empty()) {
    mLength = new float[mSize];
    memcpy(mLength,&length[0],mSize*sizeof(float));
  }

  return mSize;
}