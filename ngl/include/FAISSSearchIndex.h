/*
 * FAISSSearchIndex.h
 *
 *  Created on: Nov 9, 2018
 *      Author: maljovec
 */

#ifndef NGL_FAISS_SEARCH_INDEX_H
#define NGL_FAISS_SEARCH_INDEX_H

#include "SearchIndex.h"
#include <faiss/Index.h>
#include "faiss/gpu/StandardGpuResources.h"

class FAISSSearchIndex : public SearchIndex
{
public:
  FAISSSearchIndex() { }
  ~FAISSSearchIndex();
  void fit(float *X, int count, int dimensionality);
  void search(int *indices, int N, int K, int *k_indices, float *distances=NULL);
  void search(int startIndex, int count, int K, int *k_indices, float *distances=NULL);
private:
    float *data;
    int D;
    int my_count;
    faiss::Index *index;
    faiss::gpu::StandardGpuResources *res;
};

#endif
