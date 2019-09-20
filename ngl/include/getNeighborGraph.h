#ifndef GETNEIGHBORGRAPH_H
#define GETNEIGHBORGRAPH_H

#include <cstring>
#include "ngl.h"


void getNeighborGraph(const char* method, 
                      ngl::ANNPointSet<float>* points,
                      int kmax, 
                      float param,
                      ngl::IndexType** edges, int* numEdges);

void getSymmetricNeighborGraph(const char* method, 
                               ngl::ANNPointSet<float>* points,
                               int kmax, 
                               float param,
                               std::vector<ngl::IndexType>* edges);

#endif
