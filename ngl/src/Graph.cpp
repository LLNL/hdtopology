/*
 * Graph.h
 *
 *  Created on: Oct 22, 2018
 *      Author: maljovec
 */

#include "Graph.h"
#include "SearchIndex.h"
#include "ngl_cuda.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <cmath>

#ifdef ENABLE_OPENMP
  #include <omp.h>
  #include <cstdio>
#endif

Graph::Graph(SearchIndex *index,
             int maxNeighbors,
             bool relaxed,
             float beta,
             float p,
             int discreteSteps,
             int querySize)
    : mMaxNeighbors(maxNeighbors),
      mRelaxed(relaxed),
      mBeta(beta),
      mLp(p),
      mDiscreteSteps(discreteSteps),
      mQuerySize(querySize)
{
    if (index != NULL)
    {
        mSearchIndex = index;
    }
    else
    {
    }
}

void Graph::build(float *X, int N, int D)
{
    mData = X;
    mCount = N;
    mDim = D;

    std::cout << "buildIndex ...." << std::endl;
    std::cout << "  mCount = " << mCount << std::endl;
    std::cout << "  mDim = " << mDim << std::endl;
    mSearchIndex->fit(mData, mCount, mDim);
    std::cout << "buildIndex done\n" << std::endl;

#ifdef ENABLE_CUDA
    size_t availableGPUMemory = nglcu::get_available_device_memory();
    fprintf(stderr, "availableGPUMemory: %ld\n", availableGPUMemory);

    if (mQuerySize < 0)
    {
        // Because we are using f32 and i32:
        int bytesPerNumber = 4;
        int k = mMaxNeighbors;
        int worstCase;

        // Worst-case upper bound limit of the number of points
        // needed in a single query
        if (mRelaxed)
        {
            // For the relaxed algorithm we need n*D storage for the
            // point locations plus n*k for the representation of the
            // input edges plus another n*k for the output edges
            // We could potentially only need one set of edges for
            // this version
            worstCase = (D + 2) * k;
        }
        else
        {
            // For the strict algorithm, if we are processing n
            // points at a time, we need the point locations of all
            // of their neigbhors, thus in the worst case, we need
            // n + n*k point locations and rows of the edge matrix.
            // the 2*k again represents the need for two versions of
            // the edge matrix. Here, we definitely need an input and
            // an output array
            worstCase = k * (D + 2) * (k + 1);
        }

        // If we are using the discrete algorithm, remember we need
        // to add the template's storage as well to the GPU
        if (mDiscreteSteps > 0)
        {
            availableGPUMemory -= mDiscreteSteps * bytesPerNumber;
        }

        int divisor = bytesPerNumber * worstCase;
        int allocation = (int)( float(availableGPUMemory) / float(divisor));
        mQuerySize = std::min(allocation, mCount);
        //mQuerySize = mCount;
    }

    /*
    if(mQuerySize>100000){
       mQuerySize = 100000;
    }
    */
#else
    if(mQuerySize<0)
        mQuerySize = 1000000; //if on CPU

#endif

    fprintf(stderr, "mQuerySize: %d   mCount: %d \n", mQuerySize, mCount);
    mChunked = mQuerySize < mCount;

    std::cout << "populate first query ... " << std::endl;
    populate();
    std::cout << "populate done" << std::endl;
    // Load up the first edge
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1 && !mIterationFinished)
    {
        advanceIteration();
    }
}

void Graph::populate()
{
    if (mChunked)
    {
        fprintf(stderr, " populate size: %d\n", mQuerySize * mMaxNeighbors); 
        mEdges = new int[mQuerySize * mMaxNeighbors];
        mDistances = new float[mQuerySize * mMaxNeighbors];
        fprintf(stderr, " done allocate populate size: %d\n", mQuerySize * mMaxNeighbors); 
        populate_chunk(0);
    }
    else
    {
        mEdges = new int[mCount * mMaxNeighbors];
        mDistances = new float[mCount * mMaxNeighbors];
        populate_whole();
    }
}

void Graph::populate_chunk(int startIndex)
{
    mRowOffset = startIndex;
    int count = std::min(mCount - startIndex, mQuerySize);
    int edgeCount = count;
    int endIndex = startIndex + count;

    std::cerr << "  Graph::populate_chunk"<< std::endl;
    std::cerr << "   startIndex: " << startIndex << std::endl;
    std::cerr << "   endIndex: " << endIndex << std::endl;

    mSearchIndex->search(mRowOffset, count, mMaxNeighbors, mEdges, mDistances);

    std::unordered_set<int> additionalIndices;
    for (int i = 0; i < count; i++)
    {
        for (int k = 0; k < mMaxNeighbors; k++)
        {
            if (mEdges[i * mMaxNeighbors + k] < startIndex || mEdges[i * mMaxNeighbors + k] >= endIndex)
            {
                additionalIndices.insert(mEdges[i * mMaxNeighbors + k]);
            }
        }
    }

    std::vector<int> indices;
    for (int i = startIndex; i < endIndex; i++)
    {
        indices.push_back(i);
    }

    if (additionalIndices.size() > 0)
    {
        int extraCount = additionalIndices.size();
        for (auto it = additionalIndices.begin(); it != additionalIndices.end(); it++)
        {
            indices.push_back(*it);
        }

        if (!mRelaxed)
        {
            int *extraEdges = new int[extraCount * mMaxNeighbors];
            // This should not be necessary, but otherwise I need to make sure every index
            // properly handles the null pointer.
            float *extraDistances = new float[extraCount * mMaxNeighbors];
            int *extraIndices = indices.data() + count;

            mSearchIndex->search(extraIndices, extraCount, mMaxNeighbors, extraEdges, extraDistances);
            // Again, delete usage of extraDistances if we can make all of the SearchIndices use the following signature:
            // mSearchIndex->search(extraIndices, extraCount, mMaxNeighbors, extraEdges, NULL);
            delete [] extraDistances;

            edgeCount = count + extraCount;
            int *allEdges = new int[edgeCount * mMaxNeighbors];

            for (int i = 0; i < count; i++)
            {
                for (int k = 0; k < mMaxNeighbors; k++)
                {
                    allEdges[i * mMaxNeighbors + k] = mEdges[i * mMaxNeighbors + k];
                }
            }
            std::unordered_set<int> uniqueIndices(indices.begin(), indices.end());
            additionalIndices.clear();
            for (int i = 0; i < extraCount; i++)
            {
                for (int k = 0; k < mMaxNeighbors; k++)
                {
                    allEdges[(i + count) * mMaxNeighbors + k] = extraEdges[i * mMaxNeighbors + k];
                    additionalIndices.insert(extraEdges[i * mMaxNeighbors + k]);
                }
            }
            delete mEdges;
            delete [] extraEdges;
            mEdges = allEdges;

            for (auto it = additionalIndices.begin(); it != additionalIndices.end(); it++)
            {
                if(uniqueIndices.find(*it) == uniqueIndices.end()) {
                    indices.push_back(*it);
                }
            }
        }
    }

    float *X = new float[indices.size() * mDim];
    for (int i = 0; i < indices.size(); i++)
    {
        for (int d = 0; d < mDim; d++)
        {
            X[i * mDim + d] = mData[indices[i] * mDim + d];
        }
    }

    if (mDiscreteSteps > 0)
    {
#ifdef ENABLE_CUDA
      nglcu::prune_discrete(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                              mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp, count);
#else
      this->prune_discrete(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                            mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp, count);
#endif
    }
    else
    {
#ifdef ENABLE_CUDA
      nglcu::prune(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp, count);
#else
      this->prune(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp, count);
#endif
    }

    delete [] X;
}

void Graph::populate_whole()
{
    mSearchIndex->search(0, mCount, mMaxNeighbors, mEdges, mDistances);

    std::cerr << "  Graph::populate_whole"<< std::endl;
    std::cerr << "   Relaxed: " << mRelaxed << std::endl;
    std::cerr << "   mDiscreteSteps: " << mDiscreteSteps << std::endl;
    if (mDiscreteSteps > 0)
    {
#ifdef ENABLE_CUDA
      nglcu::prune_discrete(mData, mEdges, NULL, mCount, mDim, mCount,
                              mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp);
#else
      this->prune_discrete(mData, mEdges, NULL, mCount, mDim, mCount,
                            mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp);
#endif
    }
    else
    {
#ifdef ENABLE_CUDA
      nglcu::prune(mData, mEdges, NULL, mCount, mDim, mCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp);
#else
      this->prune(mData, mEdges, NULL, mCount, mDim, mCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp);
#endif
    }
}

void Graph::restart_iteration()
{
    mIterationFinished = false;
    mCurrentCol = 0;
    mCurrentRow = 0;
    // Load up the first edge
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1 && !mIterationFinished)
    {
        advanceIteration();
    }
}

Edge Graph::next()
{
    Edge e;
    if (mIterationFinished)
    {
        // Set up the next round of iteration
        mIterationFinished = false;
        e.indices[0] = -1;
        e.indices[1] = -1;
        e.distance = 0;
        return e;
    }
    int currentIndex = (mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol;
    e.distance = mDistances[currentIndex];
    e.indices[0] = mEdges[currentIndex];
    e.indices[1] = mCurrentRow;
    advanceIteration();
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1)
    {
        advanceIteration();
    }

    return e;
}

void Graph::advanceIteration()
{
    mCurrentCol++;
    if (mCurrentCol >= mMaxNeighbors)
    {
        mCurrentCol = 0;
        mCurrentRow++;
        if (mCurrentRow >= mCount)
        {
            mCurrentRow = 0;
            mIterationFinished = true;
        }
        // If we have changed rows, let's ensure we don't need to run
        // another query
        if (mChunked)
        {
            if (mCurrentRow - mRowOffset >= mQuerySize || mCurrentRow == 0)
            {
                populate_chunk(mCurrentRow);
            }
        }
    }
}

Graph::~Graph()
{
    if (mSelfConstructedIndex)
    {
        delete mSearchIndex;
    }
    delete mEdges;
    delete mDistances;
}


//////////
void Graph::map_indices(int *matrix, int *map, int M, int N){
  for (int row = 0; row < M; row ++) {
    for (int col = 0; col < N; col ++) {
      if(matrix[row*N+col] != -1) {
        matrix[row*N+col] = map[matrix[row*N+col]];
      }
    }
  }
}

void Graph::prune_discrete(float *X, int *edges, int *indices, int N, int D, int M, int K,
                    float *erTemplate, int steps, bool relaxed, float beta,
                    float p, int count)
{


}

void Graph::prune(float *X, int *edges, int *indices, int N, int D, int M, int K,
           bool relaxed, float beta, float lp, int count)
{
  if (count < 0) {
      count = N;
  }

  // std::cerr << "    N = "<< N << std::endl;
  // std::cerr << "    D = "<< D << std::endl;
  // std::cerr << "    M = "<< M << std::endl;
  // std::cerr << "    K = "<< K << std::endl;
  // std::cerr << "    relaxed = "<< relaxed << std::endl;
  // std::cerr << "    count = "<< count << std::endl;
  // std::cerr << "    beta = "<< beta << std::endl;
  // std::cerr << "    lp = "<< lp << std::endl;

  int *edgesOut = (int*)malloc(count*K*sizeof(int));
  memcpy(edgesOut, edges, count*K*sizeof(int));

  //map gloabl index to local index
  if (indices != NULL) {
      int *map_d;
      int i;

      int max_index = 0;
      for(i = 0; i < N; i++) {
          if(indices[i] > max_index) {
              max_index = indices[i];
          }
      }
      // std::cerr << "    max_index = "<< max_index << std::endl;
      map_d = (int*)malloc(max_index*sizeof(int));
      for(i = 0; i < N; i++) {
          map_d[indices[i]] = i;
      }
      this->map_indices(edges, map_d, M, K);
      this->map_indices(edgesOut, map_d, M, K);

      delete map_d;
      // std::cerr << "    mappedIndex ... " << std::endl;
  }

  std::cerr << "    begin purning ... " << std::endl;

  if(relaxed){
    int i;

    std::cerr << "      relax graph ... " << std::endl;

#ifdef ENABLE_OPENMP
   #pragma omp parallel
#endif
    {
    float *p, *q, *r;

    float pq[50] = {};
    float pr[50] = {};

    int  j, k, k2, d, n;
    float t;

    float length_squared;
    float squared_distance_to_edge;
    float minimum_allowable_distance;

    ////////////////////////////////////////////////////////////
    float xC, yC, radius, y;
    ////////////////////////////////////////////////////////////
#ifdef ENABLE_OPENMP
    #pragma omp for
#endif
    for (i = 0; i < count; i++) {
        for (k = 0; k < K; k++) {
            p = &(X[D*i]);
            j = edges[K*i+k];
            q = &(X[D*j]);

            length_squared = 0;
            for(d = 0; d < D; d++) {
                pq[d] = p[d] - q[d];
                length_squared += pq[d]*pq[d];
            }
            // A point should not be connected to itself
            if(length_squared == 0) {
                edgesOut[K*i+k] = -1;
                continue;
            }

            // This loop presumes that all nearer neighbors have
            // already been processed
            for(k2 = 0; k2 < k; k2++) {
                n = edgesOut[K*i+k2];
                if (n == -1){
                    continue;
                }
                r = &(X[D*n]);

                // t is the parameterization of the projection of pr onto pq
                // In layman's terms, this is the length of the shadow pr casts onto pq
                t = 0;
                for(d = 0; d < D; d++) {
                    pr[d] = p[d] - r[d];
                    t += pr[d]*pq[d];
                }

                t /= length_squared;

                if (t > 0 && t < 1) {
                    squared_distance_to_edge = 0;
                    for(d = 0; d < D; d++) {
                        squared_distance_to_edge += (pr[d] - pq[d]*t)*(pr[d] - pq[d]*t);
                    }

                    ////////////////////////////////////////////////////////////
                    // ported from python function, can possibly be improved
                    // in terms of performance
                    xC = 0;
                    yC = 0;

                    if (beta <= 1) {
                        radius = 1. / beta;
                        yC = powf(powf(radius, lp) - 1, 1. / lp);
                    }
                    else {
                        radius = beta;
                        xC = 1. - beta;
                    }
                    t = fabs(2*t-1);
                    y = powf(powf(radius, lp) - powf(t-xC, lp), 1. / lp) - yC;
                    minimum_allowable_distance = 0.5*y*sqrt(length_squared);

                    //////////////////////////////////////////////////////////
                    if(sqrt(squared_distance_to_edge) < minimum_allowable_distance) {
                        edgesOut[K*i+k] = -1;
                        break;
                    }
                }
            }
        }
    }
    }
  }else{

#ifdef ENABLE_OPENMP
   #pragma omp parallel
#endif
    {
    float *p, *q, *r;

    float pq[50] = {};
    float pr[50] = {};

    int i, j, k, k2, d, n;
    float t;

    float length_squared;
    float squared_distance_to_edge;
    float minimum_allowable_distance;

    ////////////////////////////////////////////////////////////
    float xC, yC, radius, y;
    ////////////////////////////////////////////////////////////

#ifdef ENABLE_OPENMP
    printf("thread num: %d\n", omp_get_num_threads() );
    #pragma omp for
#endif
///////////// old order is K first /////////////
    for (i = 0; i < count; i++) {
        for (k = 0; k < K; k++) {
            p = &(X[D*i]);
            j = edges[K*i+k];
            q = &(X[D*j]);

            length_squared = 0;
            for(d = 0; d < D; d++) {
                pq[d] = p[d] - q[d];
                length_squared += pq[d]*pq[d];
            }
            // A point should not be connected to itself
            if(length_squared == 0) {
                edgesOut[K*i+k] = -1;
                continue;
            }

            for(k2 = 0; k2 < 2*K; k2++) {
                n = (k2 < K) ? edges[K*i+k2] : edges[K*j+(k2-K)];
                r = &(X[D*n]);

                // t is the parameterization of the projection of pr onto pq
                // In layman's terms, this is the length of the shadow pr casts onto pq
                t = 0;
                for(d = 0; d < D; d++) {
                    pr[d] = p[d] - r[d];
                    t += pr[d]*pq[d];
                }

                t /= length_squared;

                if (t > 0 && t < 1) {
                    squared_distance_to_edge = 0;
                    for(d = 0; d < D; d++) {
                        squared_distance_to_edge += (pr[d] - pq[d]*t)*(pr[d] - pq[d]*t);
                    }

                    ////////////////////////////////////////////////////////////
                    // ported from python function, can possibly be improved
                    // in terms of performance
                    xC = 0;
                    yC = 0;

                    if (beta <= 1) {
                        radius = 1. / beta;
                        yC = powf(powf(radius, lp) - 1, 1. / lp);
                    }
                    else {
                        radius = beta;
                        xC = 1. - beta;
                    }
                    t = fabs(2*t-1);
                    y = powf(powf(radius, lp) - powf(t-xC, lp), 1. / lp) - yC;
                    minimum_allowable_distance = 0.5*y*sqrt(length_squared);

                    //////////////////////////////////////////////////////////
                    if(sqrt(squared_distance_to_edge) < minimum_allowable_distance) {
                        edgesOut[K*i+k] = -1;
                        break;
                    }
                  }
              }
            }
        }
    }
    }
    std::cerr << "    end purning " << std::endl;

    //unmap indices
    if (indices != NULL) {
      this->map_indices(edgesOut, indices, M, K);
    }

    memcpy(edges, edgesOut, count*K*sizeof(int));
    delete edgesOut;
}
