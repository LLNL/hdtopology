/*
 * FLANNSearchIndex.h
 *
 *  Created on: Nov 13, 2018
 *      Author: maljovec
 */

#ifndef NGL_FLANN_SEARCH_INDEX_H
#define NGL_FLANN_SEARCH_INDEX_H

#include "SearchIndex.h"
#include "flann/flann.h"

class FLANNSearchIndex : public SearchIndex
{
public:
  FLANNSearchIndex() {}
  ~FLANNSearchIndex();
  void fit(float *X, int N, int D);
  void search(int *indices, int N, int K, int *k_indices, float *distances=NULL);
  void search(int startIndex, int count, int K, int *k_indices, float *distances=NULL);
private:
    int D;
    int threadNum;
    Matrix<float> *dataset;
    flann::Index<flann::L2<float> > *index;
};

#endif
