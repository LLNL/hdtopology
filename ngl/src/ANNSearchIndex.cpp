/*
 * ANNSearchIndex.cpp
 *
 *  Created on: Oct 23, 2018
 *      Author: maljovec
 */

#include <ANN/ANN.h>
#include "ANNSearchIndex.h"

void ANNSearchIndex::fit(float *X, int N, int D)
{
    data = annAllocPts(N, D);
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < D; k++)
        {
            data[i][k] = X[i*D+k];
        }
    }
    kdTree = new ANNkd_tree(data, N, D);
}

void ANNSearchIndex::search(int *indices, int N, int K, int *k_indices, float *distances)
{
    ANNidxArray nnIdx = new ANNidx[K];
    ANNdistArray dists = new ANNdist[K];
    for (int i = 0; i < N; i++)
    {
        int id = indices[i];
        kdTree->annkSearch(data[id], K, nnIdx, dists, epsilon);
        for (int k = 0; k < K; k++)
        {
            int idx = i * K + k;
            k_indices[idx] = nnIdx[k];
            if (distances != NULL)
            {
                distances[idx] = sqrt(dists[k]);
            }
        }
    }
    delete nnIdx;
    delete dists;
}

void ANNSearchIndex::search(int startIndex, int count, int K, int *k_indices, float *distances)
{
    ANNidxArray nnIdx = new ANNidx[K];
    ANNdistArray dists = new ANNdist[K];
    for (int i = startIndex; i < startIndex + count; i++)
    {
        kdTree->annkSearch(data[i], K, nnIdx, dists, epsilon);
        for (int k = 0; k < K; k++)
        {
            int idx = (i - startIndex) * K + k;
            k_indices[idx] = nnIdx[k];
            if (distances != NULL)
            {
                distances[idx] = sqrt(dists[k]);
            }
        }
    }
    delete nnIdx;
    delete dists;
}

ANNSearchIndex::~ANNSearchIndex()
{
    annDeallocPts(data);
    delete kdTree;
}
