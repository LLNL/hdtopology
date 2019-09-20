/*
 * FAISSSearchIndex.cpp
 *
 *  Created on: Nov 9, 2018
 *      Author: maljovec
 */

#include <faiss/IndexFlat.h>
#include "faiss/gpu/StandardGpuResources.h"
#include <faiss/gpu/GpuIndexFlat.h>
// #include <faiss/gpu/GpuIndexIVFFlat.h>

#include "FAISSSearchIndex.h"

void FAISSSearchIndex::fit(float *X, int count, int dimensionality)
{
    data = X;
    D = dimensionality;
    my_count = count;

    // index = new faiss::IndexFlatL2(D);

    // // or
    // res = new faiss::gpu::StandardGpuResources();
    // index = new faiss::gpu::GpuIndexFlatL2(res, D);

    // // or

    // int nlist = 100;
    // index = new faiss::gpu::GpuIndexIVFFlat(res, D, nlist, faiss::METRIC_L2);
    // index->train(count, data);


    // index->add(count, data);
}

void FAISSSearchIndex::search(int *indices, int N, int K, int *k_indices, float *distances)
{
    res = new faiss::gpu::StandardGpuResources();
    index = new faiss::gpu::GpuIndexFlatL2(res, D);
    index->add(my_count, data);

    float *xq = new float[N*D];
    long int *k_long_indices = new long int[N*K];
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            xq[i*D+d] = data[indices[i]*D+d];
        }
        for (int k = 0; k < K; k++) {
            k_long_indices[i*K+k] = (long int)k_indices[i*K+k];
        }
    }
    index->search(N, xq, K, distances, k_long_indices);

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            k_indices[i*K+k] = (int)k_long_indices[i*K+k];
        }
    }
    delete xq;
    delete k_long_indices;

    delete index;
    delete res;
}

void FAISSSearchIndex::search(int startIndex, int count, int K, int *k_indices, float *distances)
{
    res = new faiss::gpu::StandardGpuResources();
    index = new faiss::gpu::GpuIndexFlatL2(res, D);
    index->add(my_count, data);

    long int *k_long_indices = new long int[count*K];
    //for (int i = 0; i < count; i++) {
    //    for (int k = 0; k < K; k++) {
    //        k_long_indices[i*K+k] = (long int)k_indices[i*K+k];
    //    }
    //}
    index->search(count, data+startIndex*D, K, distances, k_long_indices);
    for (int i = 0; i < count; i++) {
        for (int k = 0; k < K; k++) {
            k_indices[i*K+k] = (int)k_long_indices[i*K+k];
        }
    }
    delete k_long_indices;

    delete index;
    delete res;
}

FAISSSearchIndex::~FAISSSearchIndex()
{
    //delete index;
    //if (res != NULL) {
    //    delete res;
    //}
}
