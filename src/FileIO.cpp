#include "FileIO.h"

std::vector<uint32_t> load_neighborhood(const char* filename)
{
  std::vector<uint32_t> edges;
  FILE* input = fopen(filename,"r");
  uint32_t e[2];
  
  while (fscanf(input,"%d %d", e,e+1) != EOF) {
    edges.push_back(e[0]);
    edges.push_back(e[1]);
  }
      
  fclose(input);
  return edges;
}

std::vector<float> load_ascii(const char* filename)
{
  std::vector<float> points;
  FILE* input = fopen(filename,"r");
  
  float f;
  while (fscanf(input,"%f",&f) != EOF) 
    points.push_back(f);

  return points;
}

std::vector<float> load_points(FILE* input)
{
  std::vector<float> points;
  
  float f;
  while (fscanf(input,"%f",&f) != EOF) 
    points.push_back(f);

  return points;
}
