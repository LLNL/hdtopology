#ifndef FILEIO_H
#define FILEIO_H

#include <cstdio>
#include <vector>
#include <stdint.h>

std::vector<uint32_t> load_neighborhood(const char* filename);

std::vector<float> load_ascii(const char* filename);

std::vector<float> load_points(FILE* input);

//std::vector<float> load_csv(const char* filename);


#endif
