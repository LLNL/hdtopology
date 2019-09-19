/*
 * SearchIndex.h
 *
 *  Created on: Oct 23, 2018
 *      Author: maljovec
 */

#ifndef NGL_SEARCH_INDEX_H
#define NGL_SEARCH_INDEX_H
#include <cstdlib>

class SearchIndex
{
public:
  SearchIndex() {}
  virtual ~SearchIndex() {}
  virtual void fit(float *X, int N, int D) {};
  virtual void search(int *indices, int N, int k, int *k_indices, float *distances=NULL) = 0;
  virtual void search(int startIndex, int count, int k, int *k_indices, float *distances=NULL) = 0;
};

#endif
