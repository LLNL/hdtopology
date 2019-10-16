/*
 * FLANNSearchIndex.cpp
 *
 *  Created on: Nov 13, 2018
 *      Author: maljovec
 */

#include "flann/flann.h"
#include "FLANNSearchIndex.h"

void FLANNSearchIndex::fit(float *X, int N, int dimensionality)
{
    this->threadNum = 0;
    D = dimensionality;
    dataset = new flann::Matrix<float>(X, N, D);
    // construct an randomized kd-tree index using 4 kd-trees
    index = new flann::Index<flann::L2<float> >(*dataset, flann::KDTreeIndexParams(4));
    index->buildIndex();
}

void FLANNSearchIndex::search(int *indices, int N, int K, int *k_indices, float *distances)
{
    float *xq = new float[N*D];
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            xq[i*D+d] = dataset->ptr()[indices[i]*D+d];
        }
    }
    flann::Matrix<float> query(xq, N, D);
    flann::Matrix<int> knn(k_indices, N, K);
    flann::Matrix<float> dists(distances, N, K);
    flann::SearchParams params;
    params.checks = 128;
    params.eps = 0;
    params.cores = this->threadNum;
    //params.matrices_in_gpu_ram = ?;

    index->knnSearch(query, knn, dists, K, params);

    delete [] xq;
}

void FLANNSearchIndex::search(int startIndex, int count, int K, int *k_indices, float *distances)
{
    flann::Matrix<float> query(dataset->ptr()+startIndex*D, count, D);
    flann::Matrix<int> indices(k_indices, count, K);
    flann::Matrix<float> dists(distances, count, K);
    flann::SearchParams params;
    params.checks = 128;
    params.eps = 0;
    params.cores = this->threadNum;
    //params.matrices_in_gpu_ram = ?;
    index->knnSearch(query, indices, dists, K, params);
}

FLANNSearchIndex::~FLANNSearchIndex()
{
    delete dataset;
    delete index;
}
